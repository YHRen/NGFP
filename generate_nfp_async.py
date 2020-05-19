## Require Python >=3.7

import itertools
import argparse
import itertools as its
import asyncio as aio
import multiprocessing as mp
from asyncio import Queue
from collections import namedtuple
from pathlib import Path, PurePath
from functools import partial
from warnings import warn

import torch
import numpy as np
from tqdm import tqdm
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from NeuralGraph.model import QSAR
from NeuralGraph.nfp import nfp_net
from NeuralGraph.util import dev, enlarge_weights

import logging
import sys

logger = logging.getLogger("asyncio")
logger.setLevel(logging.DEBUG)
h1 = logging.StreamHandler(sys.stderr)
h2 = logging.FileHandler(filename="/tmp/asyncio_debug.log")
logger.addHandler(h1)
logger.addHandler(h2)


def try_load_net(model_file=None):
    if model_file is not None:
        model_file = Path(model_file)
        if model_file.exists() and model_file.is_file():
            net = torch.load(args.model, map_location=dev)
        else:
            raise FileNotFoundError
    else:
        net = nfp_net(pretrained=True, protein="Mpro", progress=True)
        if False:  # random large weights
            net = QSAR(hid_dim=128, n_class=1, max_degree=6)
            enlarge_weights(net, -1e4, 1e4)
    return net.to(dev)


def canonical_line_parser(line, **kwargs):
    """
        <three_letter_dataset_short_name>, <molecule_ID_name>, <smiles>
    """
    data_name, mol_name, smiles = line.split(',')
    smiles = smiles.strip('\n')
    return data_name, mol_name, smiles


async def is_valid_smile_for_NFP(sml, max_degree=6):
    """
        NFP requires a valid smile string. 
    """
    logger.debug(f"validating smile: {sml}")
    try:
        mol = Chem.MolFromSmiles(sml)
        atoms = mol.GetAtoms()
    except:
        warn(f"Not a valid SMILE: {sml}")
        return False

    for atom in atoms:
        if atom.GetDegree() >= max_degree:
            warn(f"larger than max degree {max_degree} {sml}")
            return False
    return True


def get_file_name(line_id, CHUNK_SZ, OUTPUT):
    if (line_id+1) < CHUNK_SZ: return OUTPUT
    res = '-'.join((OUTPUT, str(line_id//CHUNK_SZ*CHUNK_SZ),
                   str(line_id//CHUNK_SZ*CHUNK_SZ+CHUNK_SZ)))
    return res+".csv"


async def read_file(fp, queue, num_workers=4):
    """ 
    fp: iteratable file pointer
    queue: async queue
    """
    tasks = []
    cache = []
    for line_id, line in enumerate(fp):
        logger.debug(f"read_file, line_id {line_id}")
        item = canonical_line_parser(line)  # (data_name, mol_name, smiles)
        logger.debug(f"read_file, item {item}")
        t = aio.create_task(is_valid_smile_for_NFP(item[2]))
        tasks.append(t)
        cache.append(item)
        if len(tasks) == num_workers:
            logger.debug("takss, {len(tasks)}")
            #mols = aio.gather(*tasks, return_exceptions=False)
            mols = await aio.gather(*tasks)
            logger.debug(f"mols, {mols}")
            for c,m in zip(cache, mols):
                logger.debug(f"c,m {c}, {m}")
                await queue.put((c,m))
            tasks = []
            cache = []

    if len(tasks) > 0:
        #mols = aio.gather(*tasks, return_exceptions=False)
        mols = await aio.gather(*tasks)
        for c, m in zip(cache, mols):
            await queue.put((c, m))

    await queue.put(((None, None, None), None))  # mark the end of the queue


async def process_cache(net, cache, pool):
    logger.debug("processing cache")
    return net.calc_nfp(cache, worker_pool=pool)


async def write_file(filename, io_buffer, fps=None):
    if fps:  # writing to output
        with open(filename, 'w') as fw:
            logger.debug(f"writing to valid file {filename}")
            logger.debug(f"fps dims {len(fps), type(fps)}")
            for fp in fps:
                logger.debug(f"fp dims, fp type:  {len(fp), type(fp)}")

            fps = np.concatenate(fps)
            logger.debug(f"len = {len(io_buffer)}, {len(io_buffer[0])}, {fps.shape}")
            logger.debug(f"{io_buffer[0]}")
            for (d_, m_, s_), f_ in zip(io_buffer, fps):
                logger.debug(f"1111{d_, m_, s_, f_}")
                fp_ = ':'.join("{:.7f}".format(x) for x in f_)
                fw.write(f"{d_},{m_},{s_},{fp_}\n")
    else:  # write invalid SMILES to missing directory
        with open(filename, 'w') as fw:
            logger.debug(f"writing to miss file {filename}")
            for d_, m_, s_ in zip(io_buffer):
                fw.write(f"{d_},{m_},{s_}\n")


async def consume(queue, pool, net, pars):
    """
    pars: parameters like: CHUNK_SZ, OUTPUT, OUTPUT_DIR, BATCH_SIZE
    """

    valid_buffer, missed_buffer = [], []
    fps, cache = [], []
    idx = 0
    io_futures = []
    logger.debug("consumer")
    for idx in itertools.count():
        msg = await queue.get()
        logger.debug(f"get from queue {msg}")

        #  reached the end of the queue
        if msg[0][0] is None and msg[0][2] is None:
            logger.debug(f"reached the end")
            if len(cache) > 0:
                fp = await aio.gather(process_cache(net, cache, pool))
                logger.debug(f"fp type {type(fp)}, fp len = {len(fp)}")
                fps.append(fp[0])
            fname = get_file_name(idx, pars.CHUNK_SZ, pars.OUTPUT)

            if len(io_futures) > 0:
                await aio.gather(*io_futures)
                io_futures = []
            t = aio.create_task(write_file(pars.OUTPUT_DIR/fname,
                                           valid_buffer, fps))
            io_futures.append(t)
            if len(missed_buffer) > 0:
                t = aio.create_task(write_file(pars.MISSING_DIR/fname,
                                               missed_buffer))
                io_futures.append(t)
            await aio.gather(*io_futures)
            return

        # got a msg, put into the buffer
        if not msg[1]:
            logger.debug(f"putting into the missed buffer msg[0]")
            missed_buffer.append(msg[0])
        else:
            logger.debug(f"putting into the valid buffer msg[0]")
            valid_buffer.append(msg[0])
            logger.debug(f"putting into the SMILE cache msg[0][2]")
            cache.append(msg[0][2])

        # time to output
        if (idx+1) % pars.CHUNK_SZ == 0:  # time to output
            logger.debug(f"time to output idx = {idx}")
            if len(cache) > 0:
                fp = await aio.gather(process_cache(net, cache, pool))
                cache = []
                logger.debug(f"fp type {type(fp)}, fp len = {len(fp)}")
                fps.append(fp[0])
            fname = get_file_name(idx, pars.CHUNK_SZ, pars.OUTPUT)
            if len(io_futures) > 0: 
                await aio.gather(*io_futures)
                io_futures = []
            t = aio.create_task(write_file(pars.OUTPUT_DIR/fname,
                                           valid_buffer, fps))
            io_futures.append(t)
            if len(missed_buffer) > 0:
                t = aio.create_task(write_file(pars.MISSING_DIR/fname,
                                               missed_buffer))
                io_futures.append(t)
            fps, missed_buffer, valid_buffer = [], [], []

        # time to process on GPU
        if len(cache) >= pars.BATCH_SIZE:
            logger.debug(f"time to put to the gpu cache = {len(cache)}")
            gpu_task = aio.create_task(process_cache(net, cache, pool))
            #  hopefully it is not blocking
            fp = await aio.gather(gpu_task)
            logger.debug(f"got fps from gpu = {type(fp)}, {len(fp)}")
            fps.append(fp[0])
            cache = []


async def main(pars, net):
    queue = Queue(maxsize=int(pars.CHUNK_SZ*1.3))
    with open(pars.INPUT, 'r') as fp:
        fp = tqdm(fp) if args.tqdm else fp
        io_task = aio.create_task(read_file(fp, queue, num_workers=4))
        consumer = aio.create_task(consume(queue, worker_pool, net, pars))

        await io_task
        await consumer


if __name__ == "__main__":
    """
    This program assumes the canonical smile inputs:
        <three_letter_dataset_short_name>, <molecule_ID_name>, <smiles>
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--input_file", help="choose the input csv file",
                        type=str, required=True)
    parser.add_argument("-o","--output_dir", help="specify the output directory",
                        type=str, required=True)
    parser.add_argument("--model", help="choose the pretrained model file for nfp\
                        method. If not specified, large random weights would\
                        be used", type=str, required=False)

    parser.add_argument("-c", "--chunk_size", help="output chunk size. \
                        default=1000000", type=int, default=1000000)
    parser.add_argument("-b", "--batch_size", help="batch size for processing \
                        through NFP", type=int, default=32)
    parser.add_argument("-n", "--num_workers", type=int, default=1,
                        help="number of workers. default 1 core.\
                        0 use all cores, ")
    parser.add_argument("--dataset_name", help="specify the stem of output\
                        files", type=str)
    parser.add_argument("--tqdm", help="use tqdm progress bar",
                        action="store_true")
    args = parser.parse_args()

    Param = namedtuple('Param', ['CHUNK_SZ', 'OUTPUT', 'INPUT', 'OUTPUT_DIR',
                                 'BATCH_SIZE', 'MISSING_DIR'])

    pars = Param(
        CHUNK_SZ=args.chunk_size,
        OUTPUT_DIR=Path(args.output_dir),
        INPUT=Path(args.input_file),
        OUTPUT=args.dataset_name,
        BATCH_SIZE=args.batch_size,
        MISSING_DIR=None
    )
    if not pars.INPUT.exists():
        raise FileNotFoundError
    if pars.OUTPUT is None:
        pars = pars._replace(OUTPUT=pars.INPUT.stem)
    if pars.OUTPUT_DIR.exists() and pars.OUTPUT_DIR.is_dir():
        pass
    else:
        warn(f"dir {str(pars.OUTPUT_DIR)} does not exists.")
        warn(f"creating {str(pars.OUTPUT_DIR)}...")
        pars.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pars = pars._replace(MISSING_DIR=pars.OUTPUT_DIR/"missing")
    pars.MISSING_DIR.mkdir(exist_ok=True)

    worker_pool = None
    if args.num_workers == 0:
        worker_pool = mp.Pool(mp.cpu_count()//2)
    elif args.num_workers > 1:
        worker_pool = mp.Pool(args.num_workers)

    net = try_load_net(args.model)
    logger.debug("after loading net")
    aio.run(main(pars, net), debug=True)
    
    if worker_pool:
        worker_pool.close()
        worker_pool.join()
