# infer_p3m_portrait_matting


## :rocket: Run with Ikomia API


```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow()    

# Add the p3m process to the workflow
det = wf.add_task(name="infer_p3m_portrait_matting", auto_connect=True)

wf.run_on(url="https://raw.githubusercontent.com/Ikomia-dev/notebooks/main/examples/img/img_portrait.jpg")

# Inspect your results
display(det.get_output(1))

```


## :black_nib: Citation

[Code source](https://github.com/JizhiziLi/P3M) 

```bibtex
@inproceedings{10.1145/3474085.3475512,
author = {Li, Jizhizi and Ma, Sihan and Zhang, Jing and Tao, Dacheng},
title = {Privacy-Preserving Portrait Matting},
year = {2021},
isbn = {9781450386517},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3474085.3475512},
doi = {10.1145/3474085.3475512},
booktitle = {Proceedings of the 29th ACM International Conference on Multimedia},
pages = {3501–3509},
numpages = {9},
keywords = {trimap, benchmark, portrait matting, deep learning, semantic segmentation, privacy-preserving},
location = {Virtual Event, China},
series = {MM '21}
}
```