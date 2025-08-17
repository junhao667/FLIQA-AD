import os
import random
from typing import List, Dict, Type
import math
from matplotlib import pyplot as plt
import torch
from torch.optim import Optimizer
import transformers
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


WEIGHTS_NAME = "pytorch_model.bin"


class Trainer:
    '''trainer for single-gpu training.
    '''
    def __init__(self, args=None):
        self.train_loss_values = []  
        self.val_loss_values = []  
        self.predicted_values = []  
        self.true_values = []  
        self.acc_mmse = [] 
        self.acc_dis = []
        # self.val_accuracy = []
        self.iter_train_loss = []
        self.iter_val_loss =[]
        self.top_5_epochs =[]
        self.all_true_values = []
        self.all_predicted_values = []
        self.mse_values = []
        self.mae_values = []
        self.rmse_values = []
        self.r_squared_values = []


    def train(self,
        model,
        dataloader,
        eval_dataloader,
        epochs: int = 1,
        scheduler: str = 'WarmupCosine',
        warmup_steps: int = 10000,
        warmup_ratio: float = 0.01,
        output_path: str = './checkpoints/vision_text_pretrain',
        optimizer_class: Type[Optimizer] = torch.optim.AdamW,
        optimizer_params : Dict[str, object]= {'lr': 2e-5},
        weight_decay: float = 0.01,
        max_grad_norm: float = 1,
        use_amp: bool = False,
        accumulation_steps: int = 1,
        plot_loss_path: str = './output/plot/loss',
        plot_eval_path: str = './output/plot/eval',
        val_regression_path: str = '/output/plot/reg_val',
        medblip: bool = False
        ):
        '''
        output_path: model save path
        checkpoint_path: model load and continue to learn path
        '''
        if not os.path.exists(plot_loss_path): 
            os.makedirs(plot_loss_path)
        self.plot_loss_path = plot_loss_path
        if not os.path.exists(plot_eval_path): 
            os.makedirs(plot_eval_path)
        self.plot_eval_path = plot_eval_path
        self.accumulation_steps = accumulation_steps
        
        if use_amp:
            from torch.cuda.amp import autocast
            scaler = torch.cuda.amp.GradScaler()

        steps_per_epoch = len(dataloader)
        num_train_steps = int((steps_per_epoch) * epochs)
        warmup_steps = math.ceil(num_train_steps * warmup_ratio) #10% of train data for warm-up

        # Prepare optimizers
        param_optimizer = list(model.named_parameters())
        if not param_optimizer:
            raise RuntimeError("Model parameters are not initialized properly or are missing.")
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
        scheduler = self._get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=num_train_steps)

        model = model.cuda()

        skip_scheduler = False
        for epoch in range(epochs):
            epoch_train_loss = 0.0 
            epoch_eval_loss =0.0
            data_iterator = iter(dataloader)

            for train_iter in range(steps_per_epoch):
                model.zero_grad()
                model.train()              
                data = next(data_iterator)

                if use_amp:
                    with autocast():
                        loss = model(data) #forward
                    loss_value = loss['loss']
                    scale_before_step = scaler.get_scale()
                    scaler.scale(loss_value).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    skip_scheduler = scaler.get_scale() != scale_before_step
                else:
                    loss = model(data)
                    loss_value = loss['loss'] / self.accumulation_steps
                    loss_value.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()

                print('Epoch[{}/{}]/Iter[{}/{}]: loss: {:.4f}'.format(epoch,epochs,train_iter,steps_per_epoch,loss_value))

                self.iter_train_loss.append(loss_value.item())
                epoch_train_loss += loss_value.item()
                optimizer.zero_grad()

                if not skip_scheduler:
                    scheduler.step()
                    
                #该epoch最后一个训练batch后进行val
                data_by_label = {
                    'Dementia': {'true_values': [], 'predicted_values': [], 'indices': []},
                    'Not demented': {'true_values': [], 'predicted_values': [], 'indices': []},
                    'mild cognitive impairment (MCI)': {'true_values': [], 'predicted_values': [], 'indices': []}
                }
                index_counter = 0
                correct_mmse = 0
                samples_mmse = 0
                correct_dis = 0
                samples_dis = 0
                if train_iter == (steps_per_epoch-1):
                    eval_data_iterator = iter(eval_dataloader)
                    num_iter = len(eval_dataloader)
                    
                    for eval_iter in range(num_iter):           
                        eval_data = next(eval_data_iterator)
                        images = eval_data['images'].cuda().half()
                        text = []
                        # question = []
                        # answer = []
                        # tq = []
                        tq_mmse = []
                        tq_disease = []
                        mmse = []
                        # Label = []
                        disease = []
                        bs = len(eval_data['reports'])
                        for b in range(bs):
                            doc = eval_data['reports'][b] 
                            if 'The diagnosis is' in doc:
                                extr_text = doc.split('The MMSE Total Score is ')[0]+'the Global CDR is'+ \
                                            doc.split('the Global CDR is')[1].split('The diagnosis is ')[0]
                                text.append(extr_text)
                                mmse_num = doc.split('The MMSE Total Score is ')[1].split(',')[0]
                                mmse.append(mmse_num)
                                label = doc.split('The diagnosis is ')[1].split('.')[0]
                                label = label.replace('AD','Dementia')
                                label = label.replace('Demented','Dementia')
                                label = label.replace('NC','Not demented')
                                label = label.replace('CN','Not demented')
                                label = label.replace('Nondemented','Not demented')
                                label = label.replace('control','Not demented')
                                label = label.replace('MCI','mild cognitive impairment (MCI)')
                                label = label.replace('Converted','Not demented')#oasis2中有该类型
                                disease.append(label)

                                question_mmse = 'What is the MMSE score for this medical image?'
                                tq_mmse.append(extr_text + f'Question: {question_mmse} Answer: ')


                                question_disease = 'What will this subject be diagnosed with?'
                                tq_disease.append(extr_text + f'Question: {question_disease} Answer: ')


                                if label in data_by_label:
                                    data_by_label[label]['true_values'].append(float(mmse_num))
                                    data_by_label[label]['indices'].append(index_counter)
                                    index_counter += 1  # Increment index for each data point
                        #loss calculate
                        model.eval()
                        with torch.no_grad():  # No gradients needed for validation
                            with autocast():
                                loss_eval = model(eval_data) #forward
                                loss_eval_value = loss_eval['loss']
                        
                        self.iter_val_loss.append(loss_eval_value.item())
                        epoch_eval_loss += loss_eval_value.item()
                        print('val: Epoch[{}/{}]/Iter[{}/{}]: loss: {:.4f}'.format(epoch,epochs,eval_iter,num_iter,loss_eval_value))
                        
                        res_mmse = model.generate({"images": images, 'text':text, 'prompt': tq_mmse}) # "images": images
                        res_disease = model.generate({"images": images, 'text':text, 'prompt': tq_disease}) # "images": images
                        
                        for i in range(bs):
                            true_answer = float(mmse[i])
                            label = disease[i]
                            try:
                                predicted_answer = float(res_mmse[i])
                                if label == 'Not demented':
                                    if abs(predicted_answer - true_answer) <= 1:
                                        correct_mmse += 1
                                else:
                                    if abs(predicted_answer - true_answer) <= 3:
                                        correct_mmse += 1
                                samples_mmse += 1
                            except ValueError:
                                predicted_answer = 0.0
                                print("error converting answer to float:", res_mmse[i])
                            if label in data_by_label:
                                    data_by_label[label]['predicted_values'].append(predicted_answer)
                            print('eval_iter[{}/{}][{}/{}] report: '.format(eval_iter,num_iter,i,bs), eval_data['reports'][i])
                            print('eval_iter[{}/{}][{}/{}] prompt: '.format(eval_iter,num_iter,i,bs), tq_mmse[i])
                            print('eval_iter[{}/{}][{}/{}] gt_answer: '.format(eval_iter,num_iter,i,bs),mmse[i])
                            print('eval_iter[{}/{}][{}/{}] answer: '.format(eval_iter,num_iter,i,bs), res_mmse[i])
                            print('-----------------------------------------------')
                        #disease validation result
                        for i in range(bs):
                            predicted_answer = res_disease[i] # This assumes the model output is directly comparable
                            true_answer = disease[i]
                            if predicted_answer.strip().lower() == true_answer.strip().lower():
                                correct_dis += 1
                            samples_dis += 1
                            print('eval_iter[{}/{}][{}/{}] report: '.format(eval_iter,num_iter,i,bs), eval_data['reports'][i])
                            print('eval_iter[{}/{}][{}/{}] prompt: '.format(eval_iter,num_iter,i,bs), tq_disease[i])
                            print('eval_iter[{}/{}][{}/{}] gt_answer: '.format(eval_iter,num_iter,i,bs), disease[i])
                            print('eval_iter[{}/{}][{}/{}] answer: '.format(eval_iter,num_iter,i,bs), res_disease[i])
                            print('-----------------------------------------------')
            #result
            acc_mmse = correct_mmse / samples_mmse if samples_mmse > 0 else 0
            self.acc_mmse.append(acc_mmse)
            acc_dis = correct_dis / samples_dis if samples_dis > 0 else 0
            self.acc_dis.append(acc_dis)
            # top-5 epochs
            total_acc =(acc_mmse+acc_dis)/2
            self.update_top_5_epochs(total_acc, epoch, model.state_dict())
            print(correct_mmse,samples_mmse,correct_dis,samples_dis,
                  "acc_mmse: {:.2f}%,acc_dis:{:.2f}%".format(acc_mmse * 100,acc_dis*100))
            # avg train loss
            avg_train_loss = epoch_train_loss / steps_per_epoch
            self.train_loss_values.append(avg_train_loss)
            avg_val_loss = epoch_eval_loss / len(eval_dataloader)
            self.val_loss_values.append(avg_val_loss)
            # epoch loss plot
            self.regression_val_all(data_by_label,val_regression_path)
            self._plot()
            self._save_ckpt(epoch,output_path,epochs)

    def regression_val_all(self,data_by_label,val_regression_path):
        if not os.path.exists(val_regression_path): 
            os.makedirs(val_regression_path)
        
        for label, data in data_by_label.items():
            self.all_true_values.extend(data['true_values'])
            self.all_predicted_values.extend(data['predicted_values'])
        with open(os.path.join(val_regression_path, 'Regression_indicator_record.txt'), "a") as file:
           
            if len(self.all_true_values) > 0 and len(self.all_predicted_values) > 0 and (len(self.all_true_values) == len(self.all_predicted_values)):

                mse = mean_squared_error(self.all_true_values, self.all_predicted_values)
                mae = mean_absolute_error(self.all_true_values, self.all_predicted_values)
                rmse = np.sqrt(mse)
                r_squared = r2_score(self.all_true_values, self.all_predicted_values)

                results_str = f"Overall - MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r_squared:.4f}"
                print(results_str)
                
                file.write(results_str + "\n")

                self.mse_values.append(mse)
                self.mae_values.append(mae)
                self.rmse_values.append(rmse)
                self.r_squared_values.append(r_squared)
            else:
               
                no_data_str = "No data available to calculate metrics."
                print(no_data_str)
                file.write(no_data_str + "\n")

        
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))  # 2x2的子图布局
        # 在第一个子图中绘制MSE
        axs[0, 0].plot(self.mse_values, label='MSE', color='blue')
        axs[0, 0].set_title('Mean Squared Error')
        axs[0, 0].set_xlabel('Epoch')
        axs[0, 0].set_ylabel('MSE')
        axs[0, 0].grid(True)

        # 在第二个子图中绘制MAE
        axs[0, 1].plot(self.mae_values, label='MAE', color='red')
        axs[0, 1].set_title('Mean Absolute Error')
        axs[0, 1].set_xlabel('Epoch')
        axs[0, 1].set_ylabel('MAE')
        axs[0, 1].grid(True)

        # 在第三个子图中绘制RMSE
        axs[1, 0].plot(self.rmse_values, label='RMSE', color='green')
        axs[1, 0].set_title('Root Mean Squared Error')
        axs[1, 0].set_xlabel('Epoch')
        axs[1, 0].set_ylabel('RMSE')
        axs[1, 0].grid(True)

        # 在第四个子图中绘制R²
        axs[1, 1].plot(self.r_squared_values, label='R²', color='purple')
        axs[1, 1].set_title('R-squared')
        axs[1, 1].set_xlabel('Epoch')
        axs[1, 1].set_ylabel('R²')
        axs[1, 1].grid(True)

        # 显示图例
        for ax in axs.flat:
            ax.legend()


        plt.tight_layout()
        fig.savefig(os.path.join(val_regression_path, 'Regression_indicator.png'))
        plt.close()

        

    def _plot(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.iter_train_loss, 'r-', label='train_loss')  
        plt.xlabel('iteration')
        plt.ylabel('Loss')
        plt.title('Training Loss Over iters')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.plot_loss_path, 'train_loss_iter.png'))

        plt.close()

         # 绘制epoch损失
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_loss_values, 'r-', label='train_loss') 
        plt.plot(self.val_loss_values, 'b-', label='val_loss')  
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(' Loss Over epoch')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.plot_loss_path, 'epoch_loss.png'))
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(self.acc_dis, 'r-', label='disease cls')
        plt.plot(self.acc_mmse, 'b-', label='mmse reg')
        plt.xlabel('Epoch')
        plt.ylabel('Acc')
        plt.title('Validation Accuracy Over epoches')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.plot_eval_path, 'acc_epoch.png'))
        plt.close()

    def _plot_regression_results(self,epoch,data_by_label):
        for label, color_info in data_by_label.items():
            plt.figure(figsize=(10, 6))
            plt.plot(color_info['indices'], color_info['true_values'], 'bo-', label='True MMSE Scores')
            plt.plot(color_info['indices'], color_info['predicted_values'], 'ro-', label='Predicted MMSE Scores')
            plt.title(f'Comparison of True vs. Predicted MMSE Scores for {label}')
            plt.xlabel('Sample Index')
            plt.ylabel('MMSE Scores')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(self.plot_eval_path, f'epoch{epoch}_{label}.png'))

    @staticmethod
    def _get_scheduler(optimizer, scheduler: str, warmup_steps: int, t_total: int):
        """
        Returns the correct learning rate scheduler. Available scheduler: constantlr, warmupconstant, warmuplinear, warmupcosine, warmupcosinewithhardrestarts
        """
        scheduler = scheduler.lower()
        if scheduler == 'constantlr':
            return transformers.get_constant_schedule(optimizer)
        elif scheduler == 'warmupconstant':
            return transformers.get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
        elif scheduler == 'warmuplinear':
            return transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        elif scheduler == 'warmupcosine':
            return transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        elif scheduler == 'warmupcosinewithhardrestarts':
            return transformers.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        else:
            raise ValueError("Unknown scheduler {}".format(scheduler))
    
    def update_top_5_epochs(self, current_accuracy, current_epoch, model_state):
        if len(self.top_5_epochs) < 10:
            self.top_5_epochs.append((current_accuracy, current_epoch, model_state))
            self.top_5_epochs.sort(reverse=True, key=lambda x: x[0])  
        elif current_accuracy > self.top_5_epochs[-1][0]:
            self.top_5_epochs[-1] = (current_accuracy, current_epoch, model_state)
            self.top_5_epochs.sort(reverse=True, key=lambda x: x[0])
        self.top_5_epochs = self.top_5_epochs[:10]  
        print('update top epochs success!')

    def _save_ckpt(self, epoch, save_dir,epochs):
        if not os.path.exists(save_dir): 
            os.makedirs(save_dir)
        if epoch == epochs-1:
            for acc, epoch, state in self.top_5_epochs:
                torch.save(state, os.path.join(save_dir, f'epoch{epoch}_acc{acc*100:.2f}%.pth'))