#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )
        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        return output

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        image_attention_mask = None

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes,
                image_attention_mask=image_attention_mask
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
        ###################### choose the mode ########################
        # mode = "original"
        mode = "look at the image tokens"
        # mode = "mask according to the distance"
        # mode = "replace the tokens"
        # mode = "ignore the tokens"
        ################################################################


        #######  get the most similar token for each image token #######
        if mode == "look at the image tokens":
            most_similar_token_ids = []
            for position_embed in inputs_embeds[0][35:35+576]:
            # calculate the similarity between the position embedding and the token embedding
                temp = self.model.embed_tokens.weight.to(position_embed.device)
                similarities = torch.nn.functional.cosine_similarity(
                    position_embed.unsqueeze(0),  # shape: (1, 4096)
                    # self.model.embed_tokens.weight,  # shape: (32000, 4096)
                    temp,
                    dim=-1
                )  # shape: (32000,)
                # find the most similar token
                most_similar_token_id = similarities.argmax().item()
                most_similar_token_ids.append(most_similar_token_id)


            ############ print the most similar tokens/texts ############
            print(most_similar_token_ids)
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained('liuhaotian/llava-v1.5-7b')
            most_similar_tokens = [tokenizer.decode(token_id) for token_id in most_similar_token_ids]
            print(most_similar_tokens)

        ################################################################


        ##################################################### mask traget text
        # # Note: inputs_embeds.size() = torch.Size([1, 626, 5120])
        # mask = torch.ones(len(most_similar_token_ids), dtype=torch.bool)
        # for i, value in enumerate(most_similar_token_ids):
        #     if value in [5333]:
        #     # if value in [30296]:  # 30296, 134, 4484
        #         mask[i] = False


        # masked_token_number = attention_mask.size(0) - attention_mask.sum().item()
        # total_image_token_number = 576 - (attention_mask.size(0) - attention_mask.sum().item())
        # total_language_token_number = attention_mask.size(0) - 576
        # with open('token_number.txt', 'a') as f:
        #     f.write(f'{masked_token_number}, {total_image_token_number}, {total_language_token_number}\n')
        # attention_mask = attention_mask.unsqueeze(0).to(inputs_embeds.device)
        #####################################################
        image_embedding = inputs_embeds[0][35:35+576]  # 576, 4096
        ##################################################### mask according to the distance
        if mode == "mask according to the distance":
        # get the embedding of ्
            target_token_id = 30296
            target_token_embedding = self.model.embed_tokens(torch.tensor([target_token_id]).to(inputs_embeds.device))
            similarities = []
            for embedding in image_embedding:
                similaritie = torch.nn.functional.cosine_similarity(
                    embedding.unsqueeze(0),
                    target_token_embedding.unsqueeze(0),
                    dim=-1
                )
                similarities.append(similaritie)
            # rank
            similarities = torch.tensor(similarities)
            _, indices = torch.sort(similarities, descending=True)  # true: mask the most similar tokens
            rate = 0.33
            # mask according to similarity
            mask = torch.tensor([1 if i not in indices[:int(rate*576)] else 0 for i in range(576)], dtype=torch.bool).to(inputs_embeds.device)
                    
            # if mask randomly
            # rate = 0
            # torch.manual_seed(33)
            # indices = torch.randperm(576)
            # mask = torch.tensor([1 if i not in indices[:int(rate*576)] else 0 for i in range(576)], dtype=torch.bool).to(inputs_embeds.device)


            attention_mask = torch.ones(inputs_embeds.size(1), dtype=torch.bool).to(inputs_embeds.device)
            attention_mask[35:35+576] = mask
            attention_mask = attention_mask.unsqueeze(0).to(inputs_embeds.device)
 
        # replace the most similar tokens with dog
        if mode == "replace the tokens":
            dog_embedding = self.model.embed_tokens(torch.tensor([11203]).to(inputs_embeds.device))  # dog
            dog_embedding = self.model.embed_tokens(torch.tensor([10435]).to(inputs_embeds.device))  # horse
            dog_embedding = self.model.embed_tokens(torch.tensor([11199]).to(inputs_embeds.device))  # bird
            
            rate = 0.5
            # for i in range(576):
            # # if i in indices[:int(rate*576)]:  # change the most similar tokens to dog
            #     if i in list(range(0, int(576*rate))):  # change first rate tokens to dog
            #         image_embedding[i] = dog_embedding
            

            # if change the bbox to dog tokens
            for x in range(24):
                for y in range(24):
                    if x >= 8 and x <=20 and y>=8 and y<=16:
                        image_embedding[y*24+x] = dog_embedding

            inputs_embeds[0][35:35+576] = image_embedding

        # if ignore the bbox

        if mode == "ignore the tokens":
            
            mask = torch.ones(24, 24, dtype=torch.bool)
            mask[6:15, 2:14] = False

            mask = mask.flatten()
            image_embedding = image_embedding[mask]

            # inputs_embeds[0][35:35+576] = image_embedding
            inputs_embeds = torch.cat((inputs_embeds[0][:35], image_embedding, inputs_embeds[0][35+576:]), 0)
            inputs_embeds = inputs_embeds.unsqueeze(0)

        #####################################################
        
        
        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
