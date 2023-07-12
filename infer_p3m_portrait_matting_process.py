# Copyright (C) 2021 Ikomia SAS
# Contact: https://www.ikomia.com
#
# This file is part of the IkomiaStudio software.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import copy
from ikomia import core, dataprocess
import torch
from infer_p3m_portrait_matting.p3m.core.network import build_model
from infer_p3m_portrait_matting.p3m.core.util import *
import os
from tqdm import tqdm
from skimage.transform import resize
from torchvision import transforms
import numpy as np


# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferP3mPortraitMattingParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        self.cuda = torch.cuda.is_available()
        self.input_size = 1024
        self.update = False

    def set_values(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.cuda = utils.strtobool(param_map["cuda"])
        self.input_size = int(param_map["input_size"])
        self.update = True

    def get_values(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = {}
        param_map["cuda"] = str(self.cuda)
        param_map["input_size"] = str(self.input_size)
        return param_map


# --------------------
# - Class which implements the process
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferP3mPortraitMatting(dataprocess.C2dImageTask):

    def __init__(self, name, param):
        dataprocess.C2dImageTask.__init__(self, name)
        # Add input/output of the process here
        self.add_output(dataprocess.CImageIO())

        # Create parameters class
        if param is None:
            self.set_param_object(InferP3mPortraitMattingParam())
        else:
            self.set_param_object(copy.deepcopy(param))

        self.model_weight_url = 'https://drive.google.com/uc?export=download&id=1smX2YQGIpzKbfwDYHAwete00a_YMwoG1'
        self.model_name = 'p3mnet_pretrained_on_p3m10k.pth'
        self.device = torch.device("cpu")
        self.model = None

    def get_progress_steps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 1

    def inference_once(self, model, scale_img):
        if torch.cuda.device_count() > 0:
            tensor_img = torch.from_numpy(scale_img.astype(
                np.float32)[:, :, :]).permute(2, 0, 1).cuda()
        else:
            tensor_img = torch.from_numpy(scale_img.astype(np.float32)[
                                          :, :, :]).permute(2, 0, 1)
        input_t = tensor_img
        input_t = input_t/255.0
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        input_t = normalize(input_t)
        input_t = input_t.unsqueeze(0)
        pred_global, pred_local, pred_fusion = model(input_t)[:3]
        pred_global = pred_global.data.cpu().numpy()
        pred_global = gen_trimap_from_segmap_e2e(pred_global)
        pred_local = pred_local.data.cpu().numpy()[0, 0, :, :]
        pred_fusion = pred_fusion.data.cpu().numpy()[0, 0, :, :]
        return pred_global, pred_local, pred_fusion

    def inference_hybrid(self, model, img, max_size):
        h, w, c = img.shape
        global_ratio = 1/2
        local_ratio = 1
        resize_h = int(h*global_ratio)
        resize_w = int(w*global_ratio)
        new_h = min(max_size, resize_h - (resize_h % 32))
        new_w = min(max_size, resize_w - (resize_w % 32))
        scale_img = resize(img, (new_h, new_w))*255.0
        pred_coutour_1, pred_retouching_1, pred_fusion_1 = self.inference_once(
            model, scale_img)
        # torch.cuda.empty_cache()
        pred_coutour_1 = resize(pred_coutour_1, (h, w))*255.0
        resize_h = int(h*local_ratio)
        resize_w = int(w*local_ratio)
        new_h = min(max_size, resize_h - (resize_h % 32))
        new_w = min(max_size, resize_w - (resize_w % 32))
        scale_img = resize(img, (new_h, new_w))*255.0
        pred_coutour_2, pred_retouching_2, pred_fusion_2 = self.inference_once(
            model, scale_img)
        # torch.cuda.empty_cache()
        pred_retouching_2 = resize(pred_retouching_2, (h, w))
        predict = get_masked_local_from_global_test(
            pred_coutour_1, pred_retouching_2)

        # Image composite
        composite = generate_composite_img(img, predict)
        composite = cv2.cvtColor(composite.astype(np.uint8), cv2.COLOR_BGR2RGB)

        # Mask
        predict = predict*255.0
        predict = cv2.resize(predict, (w, h), interpolation=cv2.INTER_LINEAR)
        return predict, composite

    def run(self):
        # Core function of your process
        # Call begin_task_run() for initialization
        self.begin_task_run()
        # Get parameters :
        param = self.get_param_object()

        # Get input:
        input = self.get_input(0)

        # Get image from input/output (numpy array):
        input_image = input.get_image()

        # Load model
        if param.update or self.model is None:
            self.device = torch.device(
                "cuda") if param.cuda else torch.device("cpu")

            # Set folder path
            model_folder = os.path.join(os.path.dirname(
                os.path.realpath(__file__)), "weights")
            model_weights = os.path.join(str(model_folder), self.model_name)

            # Download model if not exist
            if not os.path.isfile(model_weights):
                os.makedirs(model_folder, exist_ok=True)
                print(f"Downloading {self.model_name} model...")
                response = requests.get(self.model_weight_url, stream=True)
                total_size = int(response.headers.get('content-length', 0))
                block_size = 1024
                progress_bar = tqdm(
                    total=total_size, unit='B', unit_scale=True)

                with open(model_weights, 'wb') as f:
                    for data in response.iter_content(block_size):
                        progress_bar.update(len(data))
                        f.write(data)
                progress_bar.close()

            self.model = build_model('r34', pretrained=False)
            # load ckpt
            ckpt = torch.load(model_weights)
            self.model.load_state_dict(ckpt['state_dict'], strict=True)

            self.model = self.model.cuda()

        # Run inference
        bin_img, portrait = self.inference_hybrid(
            self.model, input_image, param.input_size)
        param.update = False

        # Set output:
        output_bin = self.get_output(0)
        output_bin.set_image(bin_img)

        output = self.get_output(1)
        output.set_image(portrait)

        # Step progress bar (Ikomia Studio):
        self.emit_step_progress()

        # Call end_task_run() to finalize process
        self.end_task_run()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferP3mPortraitMattingFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "infer_p3m_portrait_matting"
        self.info.short_description = "Inference of Privacy-Preserving Portrait Matting (P3M)"
        self.info.description = "This algorithm proposes inference with Privacy-Preserving Portrait Matting (P3M) model."
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Background"
        self.info.version = "1.0.0"
        self.info.icon_path = "icons/icon.png"
        self.info.authors = "Li, Jizhizi and Ma, Sihan and Zhang, Jing and Tao, Dacheng"
        self.info.article = "Privacy-Preserving Portrait Matting"
        self.info.journal = "Association for Computing Machinery"
        self.info.year = 2021
        self.info.license = "MIT License"
        # URL of documentation
        self.info.documentation_link = "https://arxiv.org/pdf/2104.14222.pdf"
        # Code source repository
        self.info.repository = "https://github.com/JizhiziLi/P3M"
        # Keywords used for search
        self.info.keywords = "Portrait matting, Privacy-preserving, Semantic segmentation, Trimap"

    def create(self, param=None):
        # Create process object
        return InferP3mPortraitMatting(self.info.name, param)
