import os
import time
import datetime

import torch
from torch.utils import data
import torch.nn.functional as F
from torchvision.utils import save_image

from model import Generator
from model import Discriminator

class HyperParamters(object):
    
    def __init__(self, 
        image_size=128,
        batch_size=3,
        num_workers=1,
        mode='train',
        n_critic=1,

        # Generator
        g_lr=0.0001,

        # Discriminator
        d_lr=0.0001,

        # Optimizer
        adam_betas=(0.5, 0.999),
        
        lambda_cls=1,
        lambda_rec=10,
        lambda_gp=10
        ):
        """Contructor"""
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mode = mode
        self.n_critic = n_critic

        # Generator hyper parameters
        self.g_lr = g_lr

        # Discriminator hyper parameters
        self.d_lr = d_lr

        # Optimizer
        self.adam_betas = adam_betas

        
        self.lambda_cls = lambda_cls
        self.lambda_rec = lambda_rec
        self.lambda_gp = lambda_gp

        

class TrainingParams(object):
    
    def __init__(self, 
        resume_iter=0,
        num_iters=1,       # 200000
        num_iters_decay=1, # 100000
        lr_update_step=1000,
        log_step=1,
        sample_step=1,
        model_save_step=1,
        sample_dir='./samples',
        model_save_dir='./model'
        ):
        """Contructor"""
        self.resume_iter = resume_iter
        self.num_iters = num_iters
        self.num_iters_decay = num_iters_decay
        self.lr_update_step = lr_update_step
        self.log_step = log_step
        self.sample_step = sample_step
        self.model_save_step = model_save_step
        self.sample_dir = sample_dir
        self.model_save_dir = model_save_dir

   

class StarGAN(object):
    """    """
    
    def __init__(self, hyper_parameters):
        """Contructor"""
        # TODO Add validation for hyper parameters
        self.h_params = hyper_parameters
        self.model_ready = False
        self.datasets = []
        self.data_loaders = []
        self.data_iterators = []
        self.classes_num = []
        self.total_classes_num = 0
        self.num_datasets = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def addDataset(self, dataset, classes_num):
        """Add Dataset"""
        # Create data loaders for each dataset
        # consider those data sets in hot vectors
        self.datasets.append(dataset)
        self.classes_num.append(classes_num)
        self.total_classes_num += classes_num

        data_loader = data.DataLoader(
            dataset=dataset,
            batch_size=self.h_params.batch_size,
            shuffle=(self.h_params.mode=='train'),
            num_workers=self.h_params.num_workers)
        self.data_loaders.append(data_loader)

        self.validateDataLoader(self.num_datasets)
        self.data_iterators.append(iter(data_loader))

        self.num_datasets += 1
        self.model_ready = False
    
    def validateDataLoader(self, datasetIndex):
        data_iter = iter(self.data_loaders[datasetIndex])
        x_real, hotOneVector = next(data_iter)
        self.validateData(datasetIndex, x_real, hotOneVector)

    def validateData(self, datasetIndex, x_real, hotOneVector):
        x_real_shape = [self.h_params.batch_size, 3, self.h_params.image_size, self.h_params.image_size]
        hotOneVector_shape = [self.h_params.batch_size, self.classes_num[datasetIndex]]
        
        if not ( (x_real.shape[0] <= x_real_shape[0]) and
                 (list(x_real.shape[1:]) == x_real_shape[1:]) ):
            raise Exception(
                "x_real.shape must be [batch_size<={}, 3, image_size={}, image_size={}]."
                " Found x_real.shape={}".format(
                    self.h_params.batch_size,
                    self.h_params.image_size,
                    self.h_params.image_size,
                    x_real.shape
                ))

        if not ( (hotOneVector.shape[0] <= hotOneVector_shape[0]) and
                (list(hotOneVector.shape[1:]) == hotOneVector_shape[1:]) ):
            raise Exception(
                "hotOneVector.shape must be [batch_size<={}, num_classes={}]."
                " Found hotOneVector.shape={}".format(
                    self.h_params.batch_size,
                    self.classes_num[datasetIndex],
                    hotOneVector.shape
                ))

        if torch.sum((hotOneVector < 0) & (hotOneVector > 1)) > 0:
            raise Exception("hotOneVector must contain values of 0s and 1s only")
                
    def next(self, datasetIndex):
        try:
            x_real, hotOneVector = next(self.data_iterators[datasetIndex])
        except StopIteration:
            self.data_iterators[datasetIndex] = iter(self.data_loaders[datasetIndex])
            x_real, hotOneVector = next(self.data_iterators[datasetIndex])
        
        self.validateData(datasetIndex, x_real, hotOneVector)
        return x_real, hotOneVector
    

    def classIndexToOneHotVector(self, class_index, num_classes):
        res = torch.zeros(class_index.shape[0], num_classes)
        res[torch.arange(class_index.shape[0]), class_index] = 1
        return res

    def datasetClassesIndeces(self, datasetIndex):
        datasetClassesStartIndex = sum(self.classes_num[0:datasetIndex])
        datasetClassesEndIndex = datasetClassesStartIndex + self.classes_num[datasetIndex]
        return datasetClassesStartIndex, datasetClassesEndIndex

    def cFromLabels(self, datasetIndex, label_org):
        batch_size = label_org.shape[0]

        zeros = torch.zeros(batch_size, sum(self.classes_num))
        datasetClassesStartIndex, datasetClassesEndIndex = self.datasetClassesIndeces(datasetIndex)
        zeros[:,datasetClassesStartIndex:datasetClassesEndIndex] = label_org

        mask = torch.zeros(batch_size, self.num_datasets)
        mask[:,datasetIndex] = 1

        c_org = torch.cat([zeros, mask], dim=1)
        return c_org

    def build_model(self):
        """ """
        # Build generator
        self.G = Generator(
            self.h_params.image_size,
            self.total_classes_num + self.num_datasets
            )
            
        # Build Discriminator
        self.D = Discriminator(
            self.h_params.image_size,
            self.total_classes_num
            ) 

        # Optimizers
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.h_params.g_lr, self.h_params.adam_betas)
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.h_params.d_lr, self.h_params.adam_betas)

        self.G.to(self.device)
        self.D.to(self.device)

        self.model_ready = True

    def classification_loss(self, logit, target):
        """Compute binary or softmax cross entropy loss."""
        # TODO Which to use ??
        return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)
        # return F.cross_entropy(logit, target)


    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def prepareSamples(self):
        sampleDatasetIndex = 0
        # Fetch images from first dataset.
        # sample_real is a list of images of size batch_size
        # sample_real.shape = [batch_size, 3, image_size, image_size]
        self.sample_real, _ = self.next(sampleDatasetIndex)
        batch_size = self.sample_real.size(0)
        self.sample_real = self.sample_real.to(self.device)

        # Create targets for each class in each dataset
        c_trg_list = []
        for datasetIndex in range(self.num_datasets):
            for i in range(self.classes_num[datasetIndex]):
                # Set the target to be label i
                target_class_index = torch.ones(batch_size, dtype=torch.long) * i
                label_trg = self.classIndexToOneHotVector(target_class_index, self.classes_num[datasetIndex])
                c_trg = self.cFromLabels(datasetIndex, label_trg)

                # c_trg is the target tensor containing label i as the target for each sample
                # c_trg.shape = [batch_size, sum(num_classes) + num_datasets]
                c_trg_list.append(c_trg.to(self.device))
        
        self.c_trg_list = c_trg_list

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def generateSamples(self, sample_dir, iterationNumber):
        with torch.no_grad():
            x_fake_list = [self.sample_real]

            for c_trg in self.c_trg_list:
                x_fake_list.append(self.G(self.sample_real, c_trg))

            x_concat = torch.cat(x_fake_list, dim=3)
            sample_path = os.path.join(sample_dir, '{}-images.jpg'.format(iterationNumber))
            save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
        print('Saved real and fake images into {}...'.format(sample_path))


    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def save_model(self, model_save_dir, resume_iter):
        G_path = os.path.join(model_save_dir, '{}-G.ckpt'.format(resume_iter))
        D_path = os.path.join(model_save_dir, '{}-D.ckpt'.format(resume_iter))
        torch.save(self.G.state_dict(), G_path)
        torch.save(self.D.state_dict(), D_path)
        print('Saved model checkpoints from step {} into {}...'.format(
            resume_iter,
            model_save_dir
            ))

    def restore_model(self, model_save_dir, resume_iter):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iter))
        G_path = os.path.join(model_save_dir, '{}-G.ckpt'.format(resume_iter))
        D_path = os.path.join(model_save_dir, '{}-D.ckpt'.format(resume_iter))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    def log(self, elapsed_time, loss_log, iteration, num_iters, datasetIndex):
        elapsed_time = str(datetime.timedelta(seconds=elapsed_time))[:-7]
        log = "Elapsed [{}], Iteration [{}/{}], Dataset [{}]".format(elapsed_time, iteration, num_iters, datasetIndex)
        for tag, value in loss_log.items():
            log += ", {}: {:.4f}".format(tag, value)
        print(log)

        # TODO
        # if self.use_tensorboard:
        #     for tag, value in loss.items():
        #         self.logger.scalar_summary(tag, value, i+1)

    def preprocessInputData(self, datasetIndex):
        # Fetch real images and labels.
        # x_real.shape = [batch_size, 3, image_size, image_size]
        x_real, hotOneVector = self.next(datasetIndex)

        # label_org.shape = [batch_size, num_classes]
        # c_org = [batch_size, sum(num_classes) + num_datasets]
        label_org = hotOneVector # self.classIndexToOneHotVector(class_index, self.classes_num[datasetIndex])
        c_org = self.cFromLabels(datasetIndex, label_org)

        # Generate target domain labels randomly.
        # label_trg.shape = [batch_size, num_classes]
        # c_trg = [batch_size, sum(num_classes) + num_datasets]
        rand_idx = torch.randperm(label_org.size(0))
        label_trg = label_org[rand_idx]
        c_trg = self.cFromLabels(datasetIndex, label_trg)

        x_real = x_real.to(self.device)             # Input images.
        c_org = c_org.to(self.device)               # Original domain labels.
        c_trg = c_trg.to(self.device)               # Target domain labels.
        label_org = label_org.to(self.device)       # Labels for computing classification loss.
        label_trg = label_trg.to(self.device)       # Labels for computing classification loss.

        return x_real, label_org, label_trg, c_org, c_trg


    def trainDiscriminator(self, datasetIndex, x_real, label_org, c_trg, loss_log):
        # Compute loss with real images.
        # out_cls are the scores for the classes.
        # out_cls.shape = [batch_size, total_num_classes]
        # out_src TODO ???
        out_src, out_cls = self.D(x_real)
        datasetClassesStartIndex, datasetClassesEndIndex = self.datasetClassesIndeces(datasetIndex)
        out_cls = out_cls[:, datasetClassesStartIndex:datasetClassesEndIndex]
        d_loss_real = - torch.mean(out_src)
        d_loss_cls = self.classification_loss(out_cls, label_org)

        # Compute loss with fake images.
        x_fake = self.G(x_real, c_trg)
        out_src, _ = self.D(x_fake.detach())
        d_loss_fake = torch.mean(out_src)
        
        # Compute loss for gradient penalty.
        alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
        x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
        out_src, _ = self.D(x_hat)
        d_loss_gp = self.gradient_penalty(out_src, x_hat)
        
        # Backward and optimize.
        d_loss = d_loss_real + d_loss_fake + self.h_params.lambda_cls * d_loss_cls + self.h_params.lambda_gp * d_loss_gp
        self.reset_grad()
        d_loss.backward()
        self.d_optimizer.step()

        # Logging.
        loss_log['D/loss_real'] = d_loss_real.item()
        loss_log['D/loss_fake'] = d_loss_fake.item()
        loss_log['D/loss_cls'] = d_loss_cls.item()
        loss_log['D/loss_gp'] = d_loss_gp.item()

    def trainGenerator(self, datasetIndex, x_real, label_trg, c_org, c_trg, loss_log):
        # Original-to-target domain.
        x_fake = self.G(x_real, c_trg)
        out_src, out_cls = self.D(x_fake)
        datasetClassesStartIndex, datasetClassesEndIndex = self.datasetClassesIndeces(datasetIndex)
        out_cls = out_cls[:, datasetClassesStartIndex:datasetClassesEndIndex]
        g_loss_fake = - torch.mean(out_src)
        g_loss_cls = self.classification_loss(out_cls, label_trg)

        # Target-to-original domain.
        x_reconst = self.G(x_fake, c_org)
        g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

        # Backward and optimize.
        g_loss = g_loss_fake + self.h_params.lambda_rec * g_loss_rec + self.h_params.lambda_cls * g_loss_cls
        self.reset_grad()
        g_loss.backward()
        self.g_optimizer.step()

        # Logging.
        loss_log['G/loss_fake'] = g_loss_fake.item()
        loss_log['G/loss_rec'] = g_loss_rec.item()
        loss_log['G/loss_cls'] = g_loss_cls.item()
        
    def train(self, train_params=TrainingParams()):
        """ """
        if not self.model_ready :
            self.build_model()
        
        self.reset_data_iterators()

        self.prepareSamples()

        # TODO Used to resume training
        start_iter = 0
        if train_params.resume_iter > 0:
            self.restore_model(train_params.model_save_dir, train_params.resume_iter)
            start_iter = train_params.resume_iter

        # Learning rate cache for decaying.
        g_lr = self.h_params.g_lr
        d_lr = self.h_params.d_lr

        print('Start training...')
        start_time = time.time()
        for i in range(start_iter, train_params.num_iters):
            for datasetIndex in range(self.num_datasets):
                loss_log = {}

                x_real, label_org, label_trg, c_org, c_trg = self.preprocessInputData(datasetIndex)

                self.trainDiscriminator(datasetIndex, x_real, label_org, c_trg, loss_log)
            
                # Train the generator once after n_critic iterations
                if (i+1) % self.h_params.n_critic == 0:
                    self.trainGenerator(datasetIndex, x_real, label_trg, c_org, c_trg, loss_log)

                # Print out training info.
                if (i+1) % train_params.log_step == 0:
                    elapsed_time = time.time() - start_time
                    self.log(elapsed_time, loss_log, i+1, train_params.num_iters, datasetIndex)
                
                # END datasets loop
            # CONT. iterations loop

            # Translate fixed images for debugging.
            if (i+1) % train_params.sample_step == 0:
                self.generateSamples(train_params.sample_dir, i+1)

            # Save model checkpoints.
            if (i+1) % train_params.model_save_step == 0:
                self.save_model(train_params.model_save_dir, i+1)

            # Decay learning rates.
            if (i+1) % train_params.lr_update_step == 0 and (i+1) > (train_params.num_iters - train_params.num_iters_decay):
                g_lr -= (self.h_params.g_lr / float(train_params.num_iters_decay))
                d_lr -= (self.h_params.d_lr / float(train_params.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

            # END iterations loop

    
    def reset_data_iterators(self):
        for datasetIndex in range(self.num_datasets):
            self.data_iterators[datasetIndex] = iter(self.data_loaders[datasetIndex])
