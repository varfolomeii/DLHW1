from shutil import copyfile
import pandas as pd
import torch
import os
from torchvision import datasets, models, transforms
import time

y_train = pd.read_csv('simple_image_classification/labels_trainval.csv')
y_train = y_train.set_index('Id')
root_dir = 'simple_image_classification'
os.mkdir(os.path.join(root_dir, 'train'))
os.mkdir(os.path.join(root_dir, 'val'))
os.mkdir(os.path.join(root_dir, 'mytest'))
os.mkdir(os.path.join(root_dir, 'mytest', 'sub'))
torch.manual_seed(42)
for i in range(200):
    os.mkdir(os.path.join(root_dir, 'train', str(i)))
    os.mkdir(os.path.join(root_dir, 'val', str(i)))
for image in os.listdir(os.path.join(root_dir, 'trainval'))[:90000]:
    copyfile('{}/{}'.format(os.path.join(root_dir, 'trainval'), image),
             '{}/{}/{}'.format(os.path.join(root_dir, 'train'), y_train.loc[image]['Category'], image))
for image in os.listdir(os.path.join(root_dir, 'trainval'))[90000:]:
    copyfile('{}/{}'.format(os.path.join(root_dir, 'trainval'), image),
             '{}/{}/{}'.format(os.path.join(root_dir, 'val'), y_train.loc[image]['Category'], image))
for image in os.listdir(os.path.join(root_dir, 'test')):
  copyfile('{}/{}'.format(os.path.join(root_dir, 'test'), image),
            '{}/{}'.format(os.path.join(root_dir, 'mytest', 'sub'), image))
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'mytest': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
image_datasets = {x: datasets.ImageFolder(os.path.join(root_dir, x),
                                          data_transforms[x]) for x in ['train', 'val', 'mytest']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, num_workers=4, shuffle=True) for x in ['train', 'val']}
dataloaders['mytest'] = torch.utils.data.DataLoader(image_datasets['mytest'], batch_size=4, num_workers=4, shuffle=False)


resnet = models.resnet18()
resnet.fc = torch.nn.Linear(resnet.fc.in_features, 200)
use_gpu = torch.cuda.is_available()
if use_gpu:
    resnet = resnet.cuda()
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

def train_model(model, criterion, optimizer, scheduler, dataloaders, num_epochs=25):
  since = time.time()

  best_model_wts = model.state_dict()
  best_acc = 0.0

  for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)

    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
      if phase == 'train':
        scheduler.step()
        model.train(True)  # Set model to training mode
      else:
        model.train(False)  # Set model to evaluate mode

      running_loss = 0.0
      running_corrects = 0

      # Iterate over data.
      for data in dataloaders[phase]:
        # get the inputs
        inputs, labels = data

        if use_gpu:
          inputs = inputs.cuda()
          labels = labels.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        # backward + optimize only if in training phase
        if phase == 'train':
          loss.backward()
          optimizer.step()

        # statistics
        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels).type(torch.float)

      epoch_loss = running_loss / dataset_sizes[phase]
      epoch_acc = running_corrects / dataset_sizes[phase]

      print('{} Loss: {:.4f} Acc: {:.4f}'.format(
        phase, epoch_loss, epoch_acc))

      # deep copy the model
      if phase == 'val' and epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model_wts = model.state_dict()

    time_elapsed = time.time() - since
    print('Elapsed {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))

  time_elapsed = time.time() - since
  print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
  print('Best val Acc: {:4f}'.format(best_acc))

  # load best model weights
  model.load_state_dict(best_model_wts)
  return model

params_to_train = resnet.parameters()
criterion = torch.nn.CrossEntropyLoss()
optimizer_ft = torch.optim.SGD(params_to_train, lr=0.001, momentum=0.9)
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_model(
    resnet, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, num_epochs=20)

torch.save(model_ft.state_dict(), 'my_model')
model_ft.eval()
predictions = []
count = 0
batch = []
test_ids = [path[0].split('/')[-1] for path in image_datasets['mytest'].imgs]
for input, _ in image_datasets['mytest']:
  if count > 0 and count % 4 == 0:
    outputs = model_ft(torch.stack(batch).cuda())
    _, preds = torch.max(outputs, -1)
    for pred in preds:
        predictions.append("{0:0>4}".format(pred.item()))
    batch = [input]
    count = 1
  else:
    count += 1
    batch.append(input)

outputs = model_ft(torch.stack(batch).cuda())
_, preds = torch.max(outputs, -1)
for pred in preds:
    predictions.append("{0:0>4}".format(pred.item()))

ans = pd.DataFrame({'Id': test_ids, 'Category': predictions})
ans = ans.set_index('Id')
ans.to_csv('labels_test.csv')