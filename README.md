
# DeepMem

DeepMem is a graph-based deep learning tool that automatically generates abstract representations for kernel objects, and recognizes the objects from raw memory dumps in a fast and robust way.

Specifically, we implement 1) a memory graph model that reconstructs the content and topology information of memory dumps, 2) a graph neural network architecture to embed the nodes in the memory graph, and 3) an object detection method that cross-validates the evidence collected from different parts of objects.

## Reference
The paper of this project can be found at [DeepMem](https://www.cs.ucr.edu/~heng/pubs/deepmem_ccs18.pdf). 
If you use this code for academic purposes, please cite it as:

``` 
@inproceedings{song2018deepmem,
  title={DeepMem: Learning Graph Neural Network Models for Fast and Robust Memory Forensic Analysis},
  author={Song, Wei and Yin, Heng and Liu, Chang and Song, Dawn},
  booktitle={Proceedings of the 2018 ACM SIGSAC Conference on Computer and Communications Security},
  pages={606--618},
  year={2018},
  organization={ACM}
}
```
## How to use it?

### 1. Set up the environment
#### Python
Version 2.6 or 2.7, but not 3.0.

```apt-get install pcregrep libpcre++-dev python-dev python -y```

#### Distorm3
```python -m pip install distorm3```

More information: [https://github.com/gdabah/distorm](https://github.com/gdabah/distorm)

#### PyCrypto
```pip install pycrypto```

More information: [https://pypi.org/project/pycrypto/](https://pypi.org/project/pycrypto/)

#### TensorFlow
```pip install tensorflow==1.3.0```

### 2. Get memory dumps
Download the memory dumps used in our experiment from: https://drive.google.com/drive/u/1/folders/1SB-20rZPmSYWUo96ZjN7ue88IzBY01V8.

Create a folder memory_dumps/ in root directory of this project, and move all memory dumps to that folder.

Or you can create your memory dumps using our tool: https://github.com/bitsecurerlab/vm_mem_dump_tool.git.

### 3. Create memory graphs
```cd create_memory_graphs/```

```python create_memory_graphs.py```

### 4. Train the model
```cd build_model```

```python train_model.py --obj_type _ETHREAD```

Currently, we support 6 kernel objects: _EPROCESS, _ETHREAD, _DRIVER_OBJECT, _FILE_OBJECT, _LDR_DATA_TABLE_ENTRY, _CM_KEY_BODY.

During training, the script will save the current model to the build_model/model/ directory after every 100 iterations.

You may stop the training when the precision and recall is good enough for that kernel object.
 
### 5. Testing the model
```cd build_model```

```python test_model.py --obj_type _ETHREAD --model_path model/_ETHREAD/20190705_142745/struct2vec_edge_type.HOP3._ETHREAD-10000```
