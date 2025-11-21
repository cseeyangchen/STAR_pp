import torch
import clip
import numpy as np
import os 
import json
import clip.simple_tokenizer

_tokenizer = clip.simple_tokenizer.SimpleTokenizer()


def get_token_embeds(model, text_token_ids):
    # text model
    text_model = model.transformer
    token_embedding = model.token_embedding
    positional_embedding = model.positional_embedding
    ln_final = model.ln_final
    # shape info
    x = token_embedding(text_token_ids)  # [batch_size, seq_len, embed_dim]
    x = x + positional_embedding[:x.size(1), :]  
    x = x.permute(1, 0, 2)  #  [seq_len, batch_size, embed_dim]

    # Transformer
    for r in text_model.resblocks:
        x = r(x)

    # [batch_size, seq_len, embed_dim]
    x = x.permute(1, 0, 2)
    # LayerNorm
    x = ln_final(x)  

    # shape: [batch_size, seq_len, embed_dim]
    return x



# load clip model
def joints_concept_feature_extract(filename, device):
    part_descriptions = []
    with open(filename) as infile:
        lines = infile.readlines()
        for ind, line in enumerate(lines):
            temp_list = line.rstrip().lstrip().split(';')
            part_descriptions.append(temp_list)
    # device = "cuda:0" if torch.cuda.is_available() else "cpu"
    clip_model, _ = clip.load('ViT-L/14@336px', device)
    clip_model.cuda(device)
    semantic_feature_dict = {}
    with torch.no_grad():
        # class semantic vector
        text_dict = {}
        num_text_aug = 6   
        for ii in range(num_text_aug):
            text_dict[ii+1] = torch.cat([clip.tokenize((ele)) for ele in part_descriptions[ii]])
            semantic_feature_dict[ii] = clip_model.float().encode_text(text_dict[ii+1].to(device))
    if not os.path.exists('semantic_feature'):
        os.makedirs('semantic_feature')
    store_path = os.path.join('semantic_feature', filename.split('/')[1].split('.')[0]+'.tar')
    torch.save(semantic_feature_dict, store_path)
    return semantic_feature_dict


# load clip model
def semantic_feature_extract(filename, device):
    part_descriptions = []
    with open(filename) as infile:
        lines = infile.readlines()
        for ind, line in enumerate(lines):
            temp_list = line.rstrip().lstrip().split(';')
            part_descriptions.append(temp_list)
    # device = "cuda:0" if torch.cuda.is_available() else "cpu"
    clip_model, _ = clip.load('ViT-L/14@336px', device)
    clip_model.cuda(device)
    semantic_feature_dict = {}
    with torch.no_grad():
        # class semantic vector
        text_dict = {}
        num_text_aug = 7   # 7
        for ii in range(num_text_aug):
            if ii == 0:
                # action name
                text_dict[ii] = torch.cat([clip.tokenize((action[ii])) for action in part_descriptions])
            else:
                text_dict[ii] = torch.cat([clip.tokenize((action[0] + ',' + action[ii]),context_length=77, truncate=True) for action in part_descriptions])
                # text_dict[ii] = torch.cat([clip.tokenize((action[ii]),context_length=77, truncate=True) for action in part_descriptions])
            # semantic_feature_dict[ii] = clip_model.float().encode_text(text_dict[ii].to(device))
            semantic_feature_dict[ii] = get_token_embeds(clip_model.float(), text_dict[ii].to(device))
    if not os.path.exists('semantic_feature'):
        os.makedirs('semantic_feature')
    store_path = os.path.join('semantic_feature', filename.split('/')[1].split('.')[0]+'_token.tar')
    torch.save(semantic_feature_dict, store_path)
    # store json files
    json_path = os.path.join('semantic_feature', filename.split('/')[1].split('.')[0]+'_tokens.json') 
    token_data = {}
    for ii in range(num_text_aug):
        for token_ids in text_dict[ii]:
            for i, token_ids in enumerate(text_dict[ii]):
                decoded_tokens = [_tokenizer.decode([id.item()]) for id in token_ids if id.item() > 0]
                token_data[f"sequence_{ii}_{i}"] = decoded_tokens
                # print(decoded_tokens)
    with open(json_path, 'w') as f:
        json.dump(token_data, f, indent=2)
    return semantic_feature_dict


if __name__ == '__main__':
    # ntu series data
    ntu_filename = 'semantics/ntu120_part_descriptions.txt'
    # pku series data
    pku_filename = 'semantics/pkuv1_part_descriptions.txt'
    device = 'cuda:0'
    for filename in [ntu_filename, pku_filename]:
        semantic_feature_extract(filename, device)
        print(f'Done: {filename}')

    # motion concept feature
    pool_filename = 'semantics/joints_concept_pool.txt'
    joints_concept_feature_extract(pool_filename, device)
    print(f'Done: {pool_filename}')










