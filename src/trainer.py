"""

Author: Wilhelm Ågren, wagren@kth.se
Last edited: 19-10-2021
"""
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

from utils import WPRINT, EPRINT, TPRINT



class PyTrainer:
    """
    what should this trainer do?
    take a model, a TrainDataloader, a ValidDataloader,
    a TestDataLoader if wants to test as well, an optimizer,
    a condition/loss-func, device, num-epochs
    """
    def __init__(self, model, 
                device='cpu', 
                problem='C',
                train=None, 
                valid=None, 
                test=None, 
                optimizer=None, 
                condition=None, 
                n_epochs=10,
                verbose=False,
                **kwargs):
        if train is None and test is None:
            raise ValueError('you must provide a dataloader for either training or testing')

        self._model = model
        self._device = device
        self._problem = problem
        self._train = train
        self._valid = valid
        self._test = test
        self._optimizer = optimizer
        self._condition = condition
        self._epochs = n_epochs
        self._verbose = verbose
        WPRINT("moving model to device", str(self), self._verbose)
        self._model.to(self._device)
        self._history = {'tloss': [], 'vloss': [], 'tacc': [], 'vacc': [], 'testingloss': [], 'testingacc': []}
    
    def __str__(self):
        return 'PyTrainer'

    def plot_classification(self, style='seaborn-talk'):
        WPRINT("plotting classification training results", str(self), self._verbose)
        plt.style.use(style)
        styles = ['-', ':']
        markers = ['.', '.']
        Y1, Y2 = ['tloss', 'vloss'], ['tacc', 'vacc']
        fig, ax1 = plt.subplots(figsize=(8, 3))
        ax2 = ax1.twinx()
        for y1, y2, style, marker in zip(Y1, Y2, styles, markers):
            ax1.plot(self._history[y1], ls=style, marker=marker, ms=7, c='tab:blue', label=y1)
            ax2.plot(self._history[y2], ls=style, marker=marker, ms=7, c='tab:orange', label=y2)
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.set_ylabel('Loss', color='tab:blue')
        ax2.tick_params(axis='y', labelcolor='tab:orange')
        ax2.set_ylabel('Accuracy [%]', color='tab:orange')
        ax1.set_xlabel('Epoch')

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1+lines2, labels1+labels2)
        # plt.tight_layout()
        plt.savefig('classification-loss-acc.png')

    def test_classification(self):
        WPRINT("testing model {} on device {}".format(str(self._model), self._device), str(self), self._verbose)
        if self._test is None:
            raise ValueError('no test dataloader provided, can\'t evaluate model')

        with torch.no_grad():
            tacc = 0
            for batch, (sample, target) in enumerate(self._train):
                sample, target = sample.to(self._device).float(), target.to(self._device).long() if self._problem == 'C' else target.to(self._device).float()
                output = self._model(sample)
                _, pred = torch.max(output.data, 1)
                tacc += (pred == target).sum().item()/output.shape[0]
                self._history['testingacc'].append(tacc/len(self._train))
            WPRINT("testing done! final result: {:.1f}%".format(100*tacc/len(self._train)), str(self), self._verbose)

    def fit(self):
        WPRINT("fitting model {} on device {} for {} epochs".format(str(self._model), self._device, self._epochs), str(self), self._verbose)
        t_start = time.time()
        self._model.train()
        for epoch in range(self._epochs):
            tloss, tacc, vloss, vacc = 0, 0, 0, 0
            for batch, (sample, target) in enumerate(self._train):
                sample, target = sample.to(self._device).float(), target.to(self._device).long() if self._problem == 'C' else target.to(self._device).float()
                self._optimizer.zero_grad()
                output = self._model(sample)
                loss = self._condition(output, target)
                loss.backward()
                self._optimizer.step()
                tloss += loss.item()
                if self._problem == 'C':
                    _, pred = torch.max(output.data, 1)
                    tacc += (pred == target).sum().item()/output.shape[0]
            if self._valid is not None:
                with torch.no_grad():
                    for batch, (sample, target) in enumerate(self._valid):
                        sample, target = sample.to(self._device).float(), target.to(self._device).long() if self._problem == 'C' else target.to(self._device).float()
                        output = self._model(sample)
                        loss = self._condition(output, target)
                        vloss += loss.item()
                        if self._problem == 'C':
                            _, pred = torch.max(output.data, 1)
                            vacc += (pred == target).sum().item()/output.shape[0]
            TPRINT(epoch, tloss/len(self._train), 100*tacc/len(self._train), vloss/len(self._valid), 100*vacc/len(self._valid), problem=self._problem)
            self._history['tloss'].append(tloss/len(self._train))
            self._history['vloss'].append(vloss/len(self._valid))
            self._history['tacc'].append(100*tacc/len(self._train))
            self._history['vacc'].append(100*vacc/len(self._valid))
        t_min, t_sec = divmod(time.time()-t_start, 60)
        WPRINT("training done! elapsed time: {}:{}".format(int(t_min), int(t_sec)), str(self), self._verbose)
        fname = '../models/{}.pth'.format(str(self._model))
        WPRINT("saving model state to {}".format(fname), str(self), self._verbose)
        # torch.save(self._model.state_dict(), fname)
        np.savez_compressed('model-history.npz', self._history)

