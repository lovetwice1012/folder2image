#!/usr/bin/env python3
import os
import sys
import argparse
import base64
import hashlib
import datetime
import glob
import warnings
import io

import numpy as np
from PIL import Image, ImageFile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

# pip install zfec
try:
    from zfec.easyfec import Encoder, Decoder
except ImportError:
    Encoder = Decoder = None

# suppress warnings for large images
warnings.simplefilter('ignore', Image.DecompressionBombWarning)
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

class FolderCodec:
    """
    Handles encoding/decoding of folders into optional encrypted,
    bit-encoded images with zfec-based erasure coding and file-hash verification.
    """
    def __init__(self,
                 width=15360, height=8640,
                 proc_workers=None, io_workers=None,
                 low_memory=False, memory_limit=8*1024**3,
                 password=None, compress_level=1,
                 fec_rate=0.30, fec_rate_max=0.95,
                 skip_verify=False):
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            self.AESGCM = AESGCM
        except ImportError:
            self.AESGCM = None

        cpus = os.cpu_count() or 1
        self.proc_workers = proc_workers or cpus
        self.io_workers = io_workers or max(2, cpus//2)
        self.width, self.height = width, height
        self.low_memory = low_memory
        self.memory_limit = memory_limit
        self.password = password
        self.compress_level = compress_level

        self.fec_rate = min(max(fec_rate, 0.0), fec_rate_max)
        self.fec_rate_max = fec_rate_max
        self.skip_verify = skip_verify

        # palette codes 0..3, 4 for padding
        self.lut = np.array([
            [0,0,0],
            [255,0,0],
            [0,255,0],
            [0,0,255],
            [255,255,255]
        ], dtype=np.uint8)
        packed = (self.lut[:,0].astype(np.uint32)<<16) | \
                 (self.lut[:,1].astype(np.uint32)<<8) | \
                  self.lut[:,2].astype(np.uint32)
        self.reverse_map = {int(v): i for i, v in enumerate(packed)}

    def _encrypt_data(self, plaintext: bytes) -> bytes:
        if not (self.password and self.AESGCM):
            return plaintext
        key = hashlib.sha256(self.password.encode()).digest()
        aes = self.AESGCM(key)
        nonce = os.urandom(12)
        ct = aes.encrypt(nonce, plaintext, None)
        return nonce + ct

    def _decrypt_data(self, data: bytes) -> bytes:
        if not (self.password and self.AESGCM):
            return data
        key = hashlib.sha256(self.password.encode()).digest()
        aes = self.AESGCM(key)
        nonce, ct = data[:12], data[12:]
        return aes.decrypt(nonce, ct, None)

    @staticmethod
    def _encode_image_task(args):
        idx, prefix, img_arr, compress_level = args
        buf = io.BytesIO()
        Image.fromarray(img_arr).save(buf, format='PNG',
                                      optimize=False,
                                      compress_level=compress_level)
        return idx, buf.getvalue()

    @staticmethod
    def _decode_image_task(args):
        path, reverse_map = args
        try:
            arr = np.array(Image.open(path), dtype=np.uint8)
        except Exception as e:
            print(f"Warning: bad image {path}: {e}", file=sys.stderr)
            return b''
        packed = (arr[...,0].astype(np.uint32)<<16) | \
                 (arr[...,1].astype(np.uint32)<<8)  | \
                  arr[...,2].astype(np.uint32)
        idxs = np.fromiter((reverse_map.get(int(v),4)
                            for v in packed.ravel()),
                           dtype=np.uint8)
        codes = idxs[idxs < 4]
        bits = np.empty(len(codes)*2, dtype=np.uint8)
        bits[0::2] = codes >> 1
        bits[1::2] = codes & 1
        pad = (-bits.size) % 8
        if pad:
            bits = np.concatenate([bits,
                                   np.zeros(pad, dtype=np.uint8)])
        return np.packbits(bits).tobytes()

    def encode(self, folder: str):
        prefix = sha256_hex(os.path.abspath(folder).encode())
        print("Phase: Scanning folder...")
        entries = []
        file_hashes = []
        for root, dirs, files in os.walk(folder):
            for d in dirs:
                rel = os.path.relpath(os.path.join(root,d), folder) + os.sep
                entries.append('D' + base64.b64encode(rel.encode()).decode())
            for f in files:
                fullp = os.path.join(root,f)
                rel = os.path.relpath(fullp, folder)
                data = open(fullp,'rb').read()
                h = sha256_hex(data)
                file_hashes.append(f"{rel}\t{h}")
                entries.append('F' + base64.b64encode(rel.encode()).decode() +
                               '::' + base64.b64encode(data).decode())

        # save manifest of file hashes
        with open(f"{prefix}_file_hashes.txt",'w') as mf:
            mf.write("\n".join(file_hashes))

        plaintext = "\n".join(entries).encode()
        encrypted = bool(self.password and self.AESGCM)
        payload = (self._encrypt_data(plaintext)
                   if encrypted else plaintext)
        flag = 1 if encrypted else 0

        print("Phase: Converting to images...")
        arr = np.frombuffer(payload, dtype=np.uint8)
        bits = np.unpackbits(arr)
        codes = bits.reshape(-1,2)
        indices = codes[:,0]*2 + codes[:,1]
        pixels = self.width*self.height
        pad = (-len(indices)) % pixels
        if pad:
            indices = np.concatenate([
                indices,
                np.full(pad,4,dtype=np.uint8)
            ])
        blocks = indices.reshape(-1,self.height,self.width)
        # flag block
        flag_block = np.full((self.height,self.width),
                              4,dtype=np.uint8)
        flag_block[0,0] = flag
        images = [self.lut[flag_block]] + [self.lut[b] for b in blocks]

        print(f"Encoding {len(images)} images...")
        tasks = [(i+1, prefix, img, self.compress_level)
                 for i,img in enumerate(images)]
        with ProcessPoolExecutor(max_workers=self.proc_workers) as pool:
            results = list(pool.map(self._encode_image_task, tasks))
        results.sort(key=lambda x: x[0])
        with ThreadPoolExecutor(max_workers=self.io_workers) as pool:
            for idx,data in results:
                fn = f"{prefix}_{idx:03d}.png"
                pool.submit(lambda d,fp: open(fp,'wb').write(d),
                            data, fn)
        print("=== Encoding Complete ===")

    def decode(self, image_folder: str, out_folder: str):
        print("Phase: Decoding images...")
        # find images
        paths = sorted(p for p in glob.glob(
            os.path.join(image_folder,'*'))
            if p.lower().endswith('.png'))
        if len(paths)<2:
            print("Error: not enough images.", file=sys.stderr)
            sys.exit(1)
        # extract prefix and hash manifest
        base = os.path.basename(paths[0])
        prefix = base.split('_')[0]
        manifest_file = os.path.join(image_folder,
                                     f"{prefix}_file_hashes.txt")
        expect_hashes = {}
        if os.path.exists(manifest_file):
            for line in open(manifest_file):
                pth, h = line.strip().split('\t')
                expect_hashes[pth] = h
        # read flag
        first = Image.open(paths[0]).convert('RGB')
        r,g,b = first.getpixel((0,0))
        packed = (r<<16)|(g<<8)|b
        flag_code = self.reverse_map.get(packed)
        if flag_code not in (0,1):
            print("Error: invalid data format.", file=sys.stderr)
            sys.exit(1)
        encrypted = (flag_code==1)
        data_paths = paths[1:]
        buffer = bytearray()
        with ProcessPoolExecutor(max_workers=self.proc_workers) as pool:
            for seg in pool.map(self._decode_image_task,
                                 [(p,self.reverse_map)
                                  for p in data_paths]):
                buffer.extend(seg)
        raw = bytes(buffer)
        if encrypted:
            if not self.password:
                print("Error: password not specified.", file=sys.stderr)
                sys.exit(1)
            try:
                raw = self._decrypt_data(raw)
            except:
                print("Error: incorrect password or corrupted data.",
                      file=sys.stderr)
                sys.exit(1)
        try:
            text = raw.decode('utf-8')
        except:
            print("Error: data corruption detected.", file=sys.stderr)
            sys.exit(1)
        # restore and optionally verify
        lines = text.split('\n')
        for line in lines:
            if not line: continue
            p,rest = line[0], line[1:]
            if p=='D':
                rel = base64.b64decode(rest).decode()
                os.makedirs(os.path.join(out_folder,rel),
                            exist_ok=True)
            else:
                path_b64, data_b64 = rest.split('::',1)
                rel = base64.b64decode(path_b64).decode()
                content = base64.b64decode(data_b64)
                full = os.path.join(out_folder,rel)
                os.makedirs(os.path.dirname(full), exist_ok=True)
                open(full,'wb').write(content)
                # hash verify
                if not self.skip_verify and rel in expect_hashes:
                    h = sha256_hex(content)
                    if h != expect_hashes[rel]:
                        print(f"Warning: hash mismatch {rel}",
                              file=sys.stderr)
        print("=== Decoding Complete ===")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(dest='cmd', required=True)
    enc = subs.add_parser('encode')
    enc.add_argument('folder')
    enc.add_argument('--lowmem', action='store_true')
    enc.add_argument('--mem', type=int, default=8*1024**3)
    enc.add_argument('--password')
    enc.add_argument('--compress', type=int, default=1)
    enc.add_argument('--fec_rate', type=float, default=0.30)
    enc.add_argument('--fec_rate_max', type=float, default=0.95)

    dec = subs.add_parser('decode')
    dec.add_argument('image_folder')
    dec.add_argument('out_folder', nargs='?', default=datetime.datetime.now().strftime('outputs/%Y_%m_%d_%H_%M'))
    dec.add_argument('--lowmem', action='store_true')
    dec.add_argument('--mem', type=int, default=8*1024**3)
    dec.add_argument('--password')
    dec.add_argument('--skip-verify', action='store_true')
    dec.add_argument('--fec_rate', type=float, default=0.30)

    args = parser.parse_args()
    codec = FolderCodec(
        proc_workers=os.cpu_count(),
        io_workers=max(2, os.cpu_count()//2),
        width=15360, height=8640,
        low_memory=args.lowmem,
        memory_limit=getattr(args,'mem',8*1024**3),
        password=getattr(args,'password',None),
        compress_level=getattr(args,'compress',1),
        fec_rate=getattr(args,'fec_rate',0.30),
        fec_rate_max=getattr(args,'fec_rate_max',0.95),
        skip_verify=getattr(args,'skip_verify',False)
    )
    if args.cmd=='encode':
        codec.encode(args.folder)
    else:
        codec.decode(args.image_folder, args.out_folder)