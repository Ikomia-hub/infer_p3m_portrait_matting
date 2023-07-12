# infer_p3m_portrait_matting


## :rocket: Run with Ikomia API


```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow()    

# Add the p3m process to the workflow
det = wf.add_task(name="infer_p3m_portrait_matting", auto_connect=True)

# Set process parameters
det.set_parameters({
    "model_name" : "resnet34", # "vitae-s" or "resnet34"
    "input_size" : "1024", # stride of 32
    "cuda" : "True"})

# Run workflow on the image
wf.run_on(url="https://raw.githubusercontent.com/Ikomia-dev/notebooks/main/examples/img/img_portrait.jpg")

# Inspect your results
display(det.get_input(0))
display(det.get_output(1))

```


## :black_nib: Citation

[Code source](https://github.com/ViTAE-Transformer/P3M-Net) 

```bibtex
@article{rethink_p3m,
  title={Rethinking Portrait Matting with Pirvacy Preserving},
  author={Ma, Sihan and Li, Jizhizi and Zhang, Jing and Zhang, He and Tao, Dacheng},
  journal={International Journal of Computer Vision},
  publisher={Springer},
  ISSN={1573-1405},
  year={2023}
}
```