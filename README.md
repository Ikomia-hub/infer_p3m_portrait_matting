<div align="center">
  <img src="https://raw.githubusercontent.com/Ikomia-hub/infer_p3m_portrait_matting/main/icons/icon.png" alt="Algorithm icon">
  <h1 align="center">infer_p3m_portrait_matting</h1>
</div>
<br />
<p align="center">
    <a href="https://github.com/Ikomia-hub/infer_p3m_portrait_matting">
        <img alt="Stars" src="https://img.shields.io/github/stars/Ikomia-hub/infer_p3m_portrait_matting">
    </a>
    <a href="https://app.ikomia.ai/hub/">
        <img alt="Website" src="https://img.shields.io/website/http/app.ikomia.ai/en.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/Ikomia-hub/infer_p3m_portrait_matting/blob/main/LICENSE.md">
        <img alt="GitHub" src="https://img.shields.io/github/license/Ikomia-hub/infer_p3m_portrait_matting.svg?color=blue">
    </a>    
    <br>
    <a href="https://discord.com/invite/82Tnw9UGGc">
        <img alt="Discord community" src="https://img.shields.io/badge/Discord-white?style=social&logo=discord">
    </a> 
</p>

This algorithm proposes inference with Privacy-Preserving Portrait Matting (P3M) model.

![Face restoration codeformer](https://github.com/ViTAE-Transformer/P3M-Net/raw/main/demo/p3m_dataset.png)

## :rocket: Use with Ikomia API

#### 1. Install Ikomia API

We strongly recommend using a virtual environment. If you're not sure where to start, we offer a tutorial [here](https://www.ikomia.ai/blog/a-step-by-step-guide-to-creating-virtual-environments-in-python).

```sh
pip install ikomia
```

#### 2. Create your workflow

```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display
# Init your workflow
wf = Workflow()    

# Add the real_esrgan algorithm
algo = wf.add_task(name = 'infer_p3m_portrait_matting', auto_connect=True)

# Run on your image  
wf.run_on(url="https://raw.githubusercontent.com/Ikomia-dev/notebooks/main/examples/img/img_portrait.jpg")

# Inspect your results
display(algo.get_input(0).get_image())
display(algo.get_output(0).get_image())
```

## :sunny: Use with Ikomia Studio

Ikomia Studio offers a friendly UI with the same features as the API.

- If you haven't started using Ikomia Studio yet, download and install it from [this page](https://www.ikomia.ai/studio).

- For additional guidance on getting started with Ikomia Studio, check out [this blog post](https://www.ikomia.ai/blog/how-to-get-started-with-ikomia-studio).

## :pencil: Set algorithm parameters
- **model_name** (str) - default 'resnet34':  Name of the model, resnet34 or vitae-s
- **input_size** (int) - default: '1024': Size of the input image (stride of 32)
- **method** (str) - default: 'HYBRID': Choice of the inference method 'HYBRID' or 'RESIZE'
- **cuda** (bool): If True, CUDA-based inference (GPU). If False, run on CPU.

```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow()    

# Add the p3m process to the workflow
algo = wf.add_task(name="infer_p3m_portrait_matting", auto_connect=True)

# Set process parameters
algo.set_parameters({
    "model_name" : "resnet34",
    "input_size" : "1024",
    "method": 'HYBRID',
    "cuda" : "True"})

# Run workflow on the image
wf.run_on(url="https://raw.githubusercontent.com/Ikomia-dev/notebooks/main/examples/img/img_portrait.jpg")

# Inspect your results
display(det.get_input(0))
display(det.get_output(1))
```

## :mag: Explore algorithm outputs

Every algorithm produces specific outputs, yet they can be explored them the same way using the Ikomia API. For a more in-depth understanding of managing algorithm outputs, please refer to the [documentation](https://ikomia-dev.github.io/python-api-documentation/advanced_guide/IO_management.html).

```python
import ikomia
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_p3m_portrait_matting", auto_connect=True)

# Run on your image  
wf.run_on(url="https://raw.githubusercontent.com/Ikomia-dev/notebooks/main/examples/img/img_portrait.jpg")

# Iterate over outputs
for output in algo.get_outputs():
    # Print information
    print(output)
    # Export it to JSON
    output.to_json()
```

