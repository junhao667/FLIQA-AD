import logging
import os
import random
import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from transformers import T5TokenizerFast
from torch.nn import functional as F

from lavis.models.blip2_models.modeling_t5 import T5Config, T5ForConditionalGeneration
from models.clip_vit import create_clip_vit_L #2D 1channel
from models.eva_vit import create_eva_vit_g
from models.blip2 import Blip2Base
class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class ADFormer(Blip2Base):

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        patch_size=32,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=False,
        train_conv1=False,
        num_query_token=32,
        t5_model="google/flan-t5-xl",
        max_txt_len=60,
        embed_dim=256,
        qformer_text_input=True,
        bert_type = "bert-base-uncased",
        # prompt_path="",
        # prompt_template=""
    ):
        super().__init__()
        self.tokenizer = self.init_tokenizer(bert_type)
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, patch_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                if '3d' not in name:  # 冻结所有3d相关的层
                    param.requires_grad = False
            print('only train 3D')
        else:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = True
            print('trian all vit')
        
        if train_conv1: # only train conv1 for CLIP
            for name, param in self.visual_encoder.named_parameters():
                if 'conv1' in name:
                    param.requires_grad = True
                    print(f"✅ will train: {name}")

        self.Qformer, self.query_tokens = self.init_Qformer(
          bert_type, num_query_token, self.visual_encoder.num_features
        )
        
        if not qformer_text_input:
            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
        else:
            self.Qformer.resize_token_embeddings(len(self.tokenizer))
    
        self.Qformer.cls = None
        
        self.t5_tokenizer = T5TokenizerFast.from_pretrained(t5_model)
        # self.t5_tokenizer = T5TokenizerFast.from_pretrained(t5_model, truncation_side='left')
        # self.t5_output_tokenizer = T5TokenizerFast.from_pretrained(t5_model, truncation_side='right')
        t5_config = T5Config.from_pretrained(t5_model)
        t5_config.dense_act_fn = "gelu"
        t5_config.output_attentions = True 
        self.t5_model = T5ForConditionalGeneration.from_pretrained(
            t5_model, config=t5_config
        )

        for name, param in self.t5_model.named_parameters():
            param.requires_grad = False
            param.data = param.data.bfloat16()

        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.qa_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        self.t5_proj = nn.Linear(self.Qformer.config.hidden_size, self.t5_model.config.hidden_size)
        self.temp = nn.Parameter(0.07 * torch.ones([]))
        self.max_txt_len = max_txt_len
        self.qformer_text_input = qformer_text_input
    
    def init_vision_encoder(
        self, model_name, img_size,patch_size, drop_path_rate, use_grad_checkpoint, precision
    ):
        assert model_name in [
            "eva_clip_g",
            "eva2_clip_L",
            "clip_L",
        ], "vit model must be eva_clip_g, eva2_clip_L or clip_L"
        if model_name == "eva_clip_g":
            visual_encoder = create_eva_vit_g(
                img_size, patch_size,drop_path_rate, use_grad_checkpoint, precision
            )
#         elif model_name == "eva2_clip_L":
#             visual_encoder = create_eva2_vit_L(
#                 img_size, drop_path_rate, use_grad_checkpoint, precision
#             )
        elif model_name == "clip_L":
            visual_encoder = create_clip_vit_L(img_size,patch_size, use_grad_checkpoint, precision)
        ln_vision = LayerNorm(visual_encoder.num_features)
        self.vit_name = model_name
        return visual_encoder, ln_vision

    def forward(self, samples):

        image = samples["images"].cuda().half()
        # prompt = random.choice(self.prompt_list)
        text = []
        # question = []
        # answer_mmse = []
        # answer_disease = []
        answer = []
        qa = []
        tq = []
        bs = len(samples['reports'])
       
        qa_combinations = [
            ('What is the MMSE score for this medical image?', 'mmse'),
            ('What will this subject be diagnosed with?', 'disease')
        ]
        for b in range(bs):
            doc = samples['reports'][b]
            mmse_num = doc.split('The MMSE Total Score is ')[1].split(',')[0]
            if mmse_num != 'nan':
                extr_text = doc.split('The MMSE Total Score is ')[0] + 'the Global CDR is' + \
                            doc.split('the Global CDR is')[1].split('The diagnosis is ')[0]
                text.append(extr_text)
                mmse_num = doc.split('The MMSE Total Score is ')[1].split(',')[0]
                # answer_mmse.append(mmse_num)

                # 获取诊断信息
                label = doc.split('The diagnosis is ')[1].split('.')[0]
                label = label.replace('AD', 'Dementia')
                label = label.replace('Demented', 'Dementia')
                label = label.replace('NC', 'Not demented')
                label = label.replace('CN', 'Not demented')
                label = label.replace('Nondemented', 'Not demented')
                label = label.replace('control', 'Not demented')
                label = label.replace('MCI', 'mild cognitive impairment (MCI)')
                label = label.replace('Converted', 'Not demented')  
                # answer_disease.append(label)


                question, answer_type = random.choice(qa_combinations)

                if answer_type == 'mmse':
                    tq.append(extr_text + f'Question: {question} Answer: ')
                    # tq.append( f'Question: {question} Answer: ')
                    qa.append(f'Question: {question} Answer: ' + mmse_num)
                    answer.append(mmse_num)
                elif answer_type == 'disease':
                    tq.append(extr_text + f'Question: {question} Answer: ')
                    # tq.append(f'Question: {question} Answer: ')
                    qa.append(f'Question: {question} Answer: ' + label)
                    answer.append(label)
            
            else:
                extr_text = doc.split('The MMSE Total Score is ')[0] + 'the Global CDR is' + \
                            doc.split('the Global CDR is')[1].split('The diagnosis is ')[0]
                text.append(extr_text)
                # mmse_num = doc.split('The MMSE Total Score is ')[1].split(',')[0]
                # answer_mmse.append(mmse_num)

               
                label = doc.split('The diagnosis is ')[1].split('.')[0]
                label = label.replace('AD', 'Dementia')
                label = label.replace('Demented', 'Dementia')
                label = label.replace('NC', 'Not demented')
                label = label.replace('CN', 'Not demented')
                label = label.replace('Nondemented', 'Not demented')
                label = label.replace('control', 'Not demented')
                label = label.replace('MCI', 'mild cognitive impairment (MCI)')
                label = label.replace('Converted', 'Not demented') 
                # answer_disease.append(label)

                tq.append(extr_text + f'What will this subject be diagnosed with? Answer: ')
                qa.append(f'What will this subject be diagnosed with? Answer: ' + label)
                answer.append(label)

    
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))#[10,344,1408]patchs=tokens=344
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)#在第一维拓展匹配img的batchsize[10.32.768],querytokens=32

        if self.qformer_text_input:
            text_Qformer = self.tokenizer(
                text,
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
            Qformer_atts = torch.cat([query_atts,text_Qformer.attention_mask],dim=1)

            query_output = self.Qformer.bert(
                text_Qformer.input_ids,
                attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
        else:
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                use_cache=True,#
                return_dict=True,
            )

        '''计算ITC loss需要得到query图像和文本,
            qa与图像的特征相似度,计算对应的token输出'''
        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
            ).to(image.device)
        text_output = self.Qformer.bert(
            text_tokens.input_ids, 
            attention_mask=text_tokens.attention_mask, 
            return_dict=True,)

        qa_tokens = self.tokenizer(
            qa,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",).to(image.device)
        qa_output = self.Qformer.bert(
            qa_tokens.input_ids, 
            attention_mask=qa_tokens.attention_mask, 
            return_dict=True,)

        image_feats = F.normalize(
            self.vision_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:]), dim=-1#对最后一个dim进行
            )
        text_feat = F.normalize(
            self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1 #由clstoken
        )
        qa_feat = F.normalize(
            self.qa_proj(qa_output.last_hidden_state[:, 0, :]), dim=-1
            )

        ##################################### ITC ########################################
        
        sim_q2t = torch.matmul(image_feats.unsqueeze(1), text_feat.unsqueeze(-1)).squeeze()
        sim_i2t, _ = sim_q2t.max(-1)
        sim_i2t = sim_i2t / self.temp

        sim_t2q = torch.matmul(text_feat.unsqueeze(1).unsqueeze(1), image_feats.permute(0, 2, 1)).squeeze()
        sim_t2i, _ = sim_t2q.max(-1)
        sim_t2i = sim_t2i / self.temp


        bs = image.size(0)
        
        targets = torch.linspace(0, bs - 1, bs, dtype=int).to(image.device)

        loss_itc = (
            F.cross_entropy(sim_i2t, targets, label_smoothing=0.1)
            + F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
        ) / 2
       
        sim_q2t = torch.matmul(image_feats.unsqueeze(1), qa_feat.unsqueeze(-1)).squeeze()
        sim_i2t, _ = sim_q2t.max(-1)
        sim_i2t = sim_i2t / self.temp

        sim_t2q = torch.matmul(qa_feat.unsqueeze(1).unsqueeze(1), image_feats.permute(0, 2, 1)).squeeze()
        sim_t2i, _ = sim_t2q.max(-1)
        sim_t2i = sim_t2i / self.temp


        bs = image.size(0)
        targets = torch.linspace(0, bs - 1, bs, dtype=int).to(image.device)

        loss_itc += (
            F.cross_entropy(sim_i2t, targets, label_smoothing=0.1)
            + F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
        ) / 2
          
        ##################################### ITC ########################################

        
        #query_output.last_hidden_state=[10,54,768],querytokens=[10,32,768]
        inputs_t5 = self.t5_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])#只取query特征
        # inputs_t5 = self.t5_proj(query_output.last_hidden_state)
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)

        with self.maybe_autocast(dtype=torch.bfloat16):
            input_tokens = self.t5_tokenizer(
                tq,
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)
            
            output_tokens = self.t5_tokenizer(
                answer,
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)

            targets = output_tokens.input_ids.masked_fill(
                output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100
            )

            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            # inputs_embeds = torch.cat([inputs_embeds,inputs_t5], dim=1)

            inputs_embeds = torch.cat([inputs_embeds,inputs_t5], dim=1)  
            encoder_atts = torch.cat([input_tokens.attention_mask,atts_t5], dim=1) 

            outputs = self.t5_model(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                decoder_attention_mask=output_tokens.attention_mask,
                output_hidden_states=True,
                return_dict=True,
                labels=targets,
            )
            loss_lm = outputs.loss
            loss=loss_itc+loss_lm
            print('loss_itc', loss_itc, 'loss_lm', outputs.loss)

            return {"loss": loss}

    
    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=30,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
        device='cuda:0',
    ):
        # import pdb;pdb.set_trace()
        prompt = samples["prompt"]
        image = samples["images"]
        text = samples['text']
        bs = image.size(0)
        #文本
        input_tokens = self.t5_tokenizer(
            prompt, 
            padding="longest", 
            return_tensors="pt").to(device)
        query_tokens = self.query_tokens.expand(bs, -1, -1)
        if self.qformer_text_input:
            # remove ocr tokens in q_former (for eval textvqa)
            # qformer_prompt = prompt
            # qformer_prompt = ['Question: ' + qp.split(' Question: ')[1] for qp in qformer_prompt]
            text_Qformer = self.tokenizer(
                text,
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
            Qformer_atts = torch.cat([query_atts,text_Qformer.attention_mask],dim=1)
        #图像

        if 'images' in samples.keys():
            with self.maybe_autocast():
                image_embeds = self.ln_vision(self.visual_encoder(image))
            # image_embeds = image_embeds.float()
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
            if self.qformer_text_input:
                query_output = self.Qformer.bert(
                    text_Qformer.input_ids,
                    attention_mask=Qformer_atts,
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )
            else:
                query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )
            
            inputs_t5 = self.t5_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])
            # inputs_t5 = self.t5_proj(query_output.last_hidden_state)
            
            atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)
            encoder_atts = torch.cat([input_tokens.attention_mask,atts_t5], dim=1)
            with self.maybe_autocast(dtype=torch.bfloat16):
                inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
                inputs_embeds = torch.cat([inputs_embeds,inputs_t5], dim=1)
        #conv
        else:
            encoder_atts = input_tokens.attention_mask
            with self.maybe_autocast(dtype=torch.bfloat16):
                inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)

        outputs = self.t5_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=encoder_atts,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            temperature=temperature,
            num_beams=num_beams,
            max_new_tokens=max_length,
            min_length=min_length,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_captions,
            return_dict_in_generate=True
        )              
        output_text = self.t5_tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
        return output_text
    