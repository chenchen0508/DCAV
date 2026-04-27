from embedding_manager import EmbeddingManager
from layer_classifier import LayerClassifier
from tqdm import tqdm
import torch
import os

class ClassifierManager:
    def __init__(self, classifier_type: str):
        self.type = classifier_type
        self.classifiers = dict()
        self.testacc = dict()
        self.test_f1 = dict()
        self.test_auroc = dict()

    def _train_classifiers(
        self, 
        X_train: EmbeddingManager,
        y_train: dict[str,list],
        lr: float = 0.01,
        n_epochs: int = 100,
        batch_size: int = 32,
    ):
        print("Training classifiers...")

        self.llm_cfg = X_train.llm_cfg

        for label_name, label_list in y_train.items():
            classifier_list = []
            for i in tqdm(range(self.llm_cfg.n_layer)):
                layer_classifier = LayerClassifier(X_train.llm_cfg, lr)
                layer_classifier.train(
                    features=X_train.layers[label_name][i],
                    labels=label_list,
                    n_epoch=n_epochs,
                    batch_size=batch_size,
                )
                classifier_list.append(layer_classifier)
            self.classifiers[label_name] = classifier_list

    def _evaluate_performance(self, X_test: EmbeddingManager, y_test: dict[str,list]):
        for label_name, label_list in y_test.items():
            test_accs = []
            test_f1s = []
            test_aurocs = []
            for i in tqdm(range(len(self.classifiers[label_name]))):
                acc, f1, auroc = self.classifiers[label_name][i].evaluate(
                    X_test = X_test.layers[label_name][i],
                    y_test = label_list
                )
                test_accs.append(acc)
                test_f1s.append(f1)
                test_aurocs.append(auroc)
            self.testacc[label_name] = test_accs
            self.test_f1[label_name] = test_f1s
            self.test_auroc[label_name] = test_aurocs
    
    def fit(
        self, 
        X_train: EmbeddingManager,
        y_train: dict[str,list],
        X_test: EmbeddingManager,
        y_test: dict[str,list],
        lr: float = 0.01,
        n_epochs: int = 100,
        batch_size: int = 32,
    ):
        self._train_classifiers(
            X_train,
            y_train,
            lr,
            n_epochs,
            batch_size,
        )

        self._evaluate_performance(
            X_test,
            y_test,
        )

        return self
    
    def save(self, relative_path: str):
        file_name = f"{self.type}_{self.llm_cfg.model_nickname}.pth"
        torch.save(self, os.path.join(relative_path, file_name))
    
    def cal_perturbation(
        self,
        embds_tensor: torch.tensor,
        label: str,
        layer: int,
        target_prob: float=0.001,
    ):
        w, b = self.classifiers[label][layer].get_weights_bias()
        logit_target = torch.log(torch.tensor(target_prob / (1 - target_prob)))
        w_norm = torch.norm(w)

        epsilon = (logit_target - b - torch.sum(embds_tensor * w, dim=1)) / w_norm
        perturbation = epsilon * w / w_norm

        return perturbation
        
    #l    
    def getCAV(
        self,
        label: str,
        layer: int,):
        w, b = self.classifiers[label][layer].get_weights_bias()
        return w
    
def load_classifier_manager(file_path: str):
    return torch.load(file_path, weights_only=False)