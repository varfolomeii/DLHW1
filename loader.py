from shutil import copyfile
import pandas as pd
import torch
import os
from torchvision import datasets, models, transforms
import time

root_dir = 'simple_image_classification'
torch.manual_seed(42)
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'mytest': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
image_datasets = {x: datasets.ImageFolder(os.path.join(root_dir, x),
                                          data_transforms[x]) for x in ['train', 'val', 'mytest']}

class_names = image_datasets['train'].classes

model_ft = torch.load('my_model')
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
        predictions.append("{0:0>4}".format(class_names[pred]))
    batch = [input]
    count = 1
  else:
    count += 1
    batch.append(input)

outputs = model_ft(torch.stack(batch).cuda())
_, preds = torch.max(outputs, -1)
for pred in preds:
    predictions.append("{0:0>4}".format(class_names[pred]))

ans = pd.DataFrame({'Id': test_ids, 'Category': predictions})
ans = ans.set_index('Id')
ans.to_csv('labels_test.csv')