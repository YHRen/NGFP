import torch
import argparse
import numpy as np
import itertools as its
import asyncio as aio
import itertools
from aio import Queue
from functools import partial
from collections import namedtuple
import multiprocessing as mp
from tqdm import tqdm
from pathlib import Path, PurePath
from warnings import warn
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from NeuralGraph.model import QSAR
from NeuralGraph.nfp import nfp_net
from NeuralGraph.util import dev, enlarge_weights

def try_load_net(model_file=None):
    if model_file is not None:
        model_file = Path(model_file)
        if model_file.exists() and model_file.is_file():
            net = torch.load(args.model, map_location=dev)
        else:
            raise FileNotFoundError
    else: 
        net = nfp_net(pretrained=True, protein="Mpro", progress=True)
        if False: # random large weights
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
        item = canonical_line_parser(line)  # (data_name, mol_name, smiles)
        t = aio.create_task(is_valid_smile_for_NFP(item[2]))
        tasks.append(t)
        cache.append(item)
        if len(tasks) == num_workers:
            mols = aio.gather(*tasks, return_exceptions=True)
            for c,m in zip(cache,mols):
                await queue.put((c,m))
            tasks = []
            cache = []

    if len(tasks) > 0:
        mols = aio.gather(*tasks, return_exceptions=True)
        for c,m in zip(cache,mols):
            await queue.put((c,m))

    await queue.put(((None, None, None), None))  # mark the end of the queue


async def process_cache(net, cache, pool):
    return net.calc_nfp(cache, worker_pool=pool)

async def write_file(pars, valid_buffer, fps=None):

async def consume(queue, pool, net, pars):
    """
    pars: parameters like: CHUNK_SZ, OUTPUT, OUTPUT_DIR, BATCH_SIZE
    """

    valid_buffer, missed_buffer = [], []
    fps, cache = [], []
    idx = 0
    io_futures = []
    for idx in itertools.count():
        msg = await queue.get()
        
        #  reached the end of the queue
        if msg[0][0] == None and msg[0][2] == None:
            if len(cache) > 0:
                fp = await aio.run(process_cache(net, cache, pool))
                fps.append(fp)
            fname = get_file_name(idx, pars.CHUNK_SZ, pars.OUTPUT)
            
            if len(io_futures) > 0:
                aio.gather(*io_futures)
                io_futures = []
            t = aio.create_task(write_file(pars.OUTPUT_DIR/fname,
                                           valid_buffer, fps))
            io_futures.append(t)
            if len(missed_buffer) > 0:
                t = aio.create_task(write_file(pars.MISSING_DIR/fname,
                                               missed_buffer))
                io_futures.append(t)
            aio.gather(*io_futures)
            return

        # got a msg
        if msg[1] == False:
            missed_buffer.append(msg[0])
        else:
            valid_buffer.append(msg[0])
            cache.append(msg[0][2])

        # time to output
        if (idx+1)%CHUNK_SZ==0:  # time to output
            if len(cache) > 0:
                fp = await aio.run(process_cache(net, cache, pool))
                cache = []
                fps.append(fp)
            fname = get_file_name(idx, pars.CHUNK_SZ, pars.OUTPUT)
            if len(io_futures) > 0: 
                aio.gather(*io_futures)
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
            fp = await aio.run(process_cache(net, cache, pool))
            cache = []
            fps.append(fp)


def oscillator(period):
    x, y = 1, period
    def f():
        nonlocal x
        z = x==0
        x = (x+1)%y
        return z
    return f



## TODO: create parameter class

class Parameter():

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

    OUTPUT_DIR = Path(args.output_dir)
    INPUT = Path(args.input_file)
    OUTPUT = args.dataset_name
    CHUNK_SZ = args.chunk_size
    if not INPUT.exists(): raise FileNotFoundError
    if OUTPUT is None: OUTPUT = INPUT.stem
    if OUTPUT_DIR.exists() and OUTPUT_DIR.is_dir(): pass
    else:
        warn(f"dir {str(OUTPUT_DIR)} does not exists.")
        warn(f"creating {str(OUTPUT_DIR)}...")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MISSING_DIR = OUTPUT_DIR/"missing"
    MISSING_DIR.mkdir(exist_ok=True)

    worker_pool = None
    if args.num_workers == 0:
        worker_pool = mp.Pool(mp.cpu_count()//2)
    elif args.num_workers > 1:
        worker_pool = mp.Pool(args.num_workers)


    ds_names, mol_names, smls, fps = [], [], [], []
    cache, missings = [], []
    net = try_load_net(args.model)
    osc = oscillator(args.batch_size)
    
    with open(INPUT, 'r') as in_f:
        fp = tqdm(in_f) if args.tqdm else in_f 
        last_line_id = 0
        for line_id, line in enumerate(fp):
            last_line_id = line_id
            if osc() and len(cache) > 0:
                # have enough strings in the batch, go through nfp.
                fps.append(net.calc_nfp(cache, worker_pool=worker_pool))
                smls.extend(cache)
                cache = []
            ds_name, mol_name, sml = canonical_line_parser(line)
            if is_valid_smile_for_NFP(sml, 6):
                ds_names.append(ds_name)
                mol_names.append(mol_name)
                cache.append(sml)
            else:
                missings.append(line)

            if (line_id+1)%CHUNK_SZ == 0:
                #output to file. for the rest in the cache
                if len(cache) > 0:
                    fps.append(net.calc_nfp(cache))
                    smls.extend(cache)
                    cache = []

                #output file
                filename = get_file_name(line_id, CHUNK_SZ, OUTPUT)
                print("filename", filename)
                # with open(OUTPUT_DIR/filename, 'w') as fw:
                #     fps = np.concatenate(fps)
                #     for d_, m_, s_, f_ in zip(ds_names, mol_names, smls, fps):
                #         fp_ = ':'.join("{:.7f}".format(x) for x in f_)
                #         fw.write(f"{d_},{m_},{s_},{fp_}\n")
                # with open(MISSING_DIR/filename, 'w') as fw:
                #     for ms_ in missings:
                #         fw.write(ms_)
                ds_names, mol_names, smls, fps, missings = [], [], [], [], []



        #for the rest of lines
        if (last_line_id+1)%CHUNK_SZ != 0:
            if last_line_id > CHUNK_SZ:
                filename = get_file_name(last_line_id, CHUNK_SZ, OUTPUT)
            else: # small dataset 
                filename = OUTPUT+".csv"
            print("last filename:", filename)
            if len(cache) > 0:
                fps.append(net.calc_nfp(cache))
                smls.extend(cache)
                cache = []
            #  with open(OUTPUT_DIR/filename, 'w') as fw:
            #      fps = np.concatenate(fps)
            #      for d_, m_, s_, f_ in zip(ds_names, mol_names, smls, fps):
            #          fp_ = ':'.join("{:.7f}".format(x) for x in f_)
            #          fw.write(f"{d_},{m_},{s_},{fp_}\n")
            #  with open(MISSING_DIR/filename, 'w') as fw:
            #      for ms_ in missings:
            #          fw.write(ms_)
        if worker_pool:
            worker_pool.close()
            worker_pool.join()
