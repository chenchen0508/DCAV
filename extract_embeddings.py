from toxic_data import prepare_jigsaw,prepare_jigsaw_balance,prepare_my_data,prepare_gcav_random,prepare_inout_data
from model_extraction import ModelExtraction
from classifier_manager import ClassifierManager
import pickle
import argparse
import torch
import torch.nn.functional as F

classifier_type = 'safety'

def extract_embds(model_nickname: str, train_ratio=0.3):

    prompt_train, y_train, prompt_test, y_test = prepare_inout_data(train_ratio=train_ratio)

    llm = ModelExtraction(model_nickname)
    
    X_train = llm.extract_embds(prompt_train)
    X_test = llm.extract_embds(prompt_test)
    pickle.dump(X_train, open(f"pickles/{model_nickname}_X_train.pkl", "wb"))
    pickle.dump(X_test, open(f"pickles/{model_nickname}_X_test.pkl", "wb"))

    pickle.dump(prompt_test, open(f"pickles/{model_nickname}_prompt_test.pkl", "wb"))
    pickle.dump(y_test, open(f"pickles/{model_nickname}_y_test.pkl", "wb"))

    clfr = ClassifierManager(classifier_type)
    clfr.fit(X_train, y_train, X_test, y_test)
    print(clfr.testacc)

    pickle.dump(clfr, open(f"pickles/{model_nickname}_clfr.pkl", "wb"))


def get_input_output_cos(model_nickname: str, train_ratio=0.3):
    llm = ModelExtraction(model_nickname)
    prompt_train, y_train, prompt_test, y_test = prepare_my_data(filepath="ToxicData/input_data.jsonl",train_ratio=train_ratio)
    
    X_train = llm.extract_embds(prompt_train)
    X_test = llm.extract_embds(prompt_test)

    clfr_in = ClassifierManager(classifier_type)
    clfr_in.fit(X_train, y_train, X_test, y_test)


    prompt_train, y_train, prompt_test, y_test = prepare_my_data(filepath="ToxicData/output_data.jsonl",train_ratio=train_ratio)
    
    X_train = llm.extract_embds(prompt_train)
    X_test = llm.extract_embds(prompt_test)

    clfr_out = ClassifierManager(classifier_type)
    clfr_out.fit(X_train, y_train, X_test, y_test)

    for i in range(32):
        cav_out = clfr_out.getCAV("toxic",i)
        cav_in = clfr_in.getCAV("toxic",i)
        v_in  = cav_in.float().squeeze(0)   # [d_model]
        v_out = cav_out.float().squeeze(0)  # [d_model]

        cos_theta = F.cosine_similarity(v_in.unsqueeze(0), v_out.unsqueeze(0)).item()
        print("layer = ", i , " cos(v_in, v_out) =", cos_theta)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='mistral-7b')
    parser.add_argument('--train_ratio','-r', type=float, default=0.5)
    args = parser.parse_args()

    model_nickname = args.model
    train_ratio = args.train_ratio

    extract_embds(model_nickname, train_ratio=train_ratio)