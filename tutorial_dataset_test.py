from tutorial_dataset import MyDataset

dataset = MyDataset('/root/autodl-tmp/coco_controlnet')
print(len(dataset))

item = dataset[1234]
jpg = item['jpg']
txt = item['txt']
hint = item['hint']
print(txt)
print(jpg.shape)
print(hint.shape)
