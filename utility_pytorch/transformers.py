from torchvision import transforms
try:
    from . import custom_transformers
except:
    import custom_transformers


train_cifar10_standard = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_cifar10_standard = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


train_cifar10_224 = transforms.Compose([
    transforms.RandomSizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_cifar10_224 = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


train_cifar10_128 = transforms.Compose([
    transforms.RandomSizedCrop(128),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_cifar10_128 = transforms.Compose([
    transforms.Scale(146),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

for i in range(0, 10):
    exec('''train_cifar10_color_0_{} = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    custom_transformers.InterpolateToGrey(0.{}),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])'''.format(i, i))
    exec('''test_cifar10_color_0_{} = transforms.Compose([
    custom_transformers.InterpolateToGrey(0.{}),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])'''.format(i, i))

for i in range(0, 10):
    exec('''train_ilsvrc_transform_color_{} = transforms.Compose([
    transforms.RandomSizedCrop(224),
    transforms.RandomHorizontalFlip(),
    custom_transformers.InterpolateToGrey(0.{}),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])'''.format(i, i))
    exec('''test_ilsvrc_transform_color_{} = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    custom_transformers.InterpolateToGrey(0.{}),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])'''.format(i, i))

for i in range(2, 33):
    exec('''train_cifar10_tile_{} = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    custom_transformers.ShuffleTiles(({}, {})),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])'''.format(i, i, i))
    exec('''test_cifar10_tile_{} = transforms.Compose([
    custom_transformers.ShuffleTiles(({}, {})),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])'''.format(i, i, i))

for i in range(2, 113):
    exec('''train_ilsvrc_transform_tile_{} = transforms.Compose([
    transforms.RandomSizedCrop(224),
    transforms.RandomHorizontalFlip(),
    custom_transformers.ShuffleTiles(({}, {})),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])'''.format(i, i, i))
    exec('''test_ilsvrc_transform_tile_{} = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])'''.format(i, i))

for i in range(1, 10):
    exec('''train_cifar10_lowpass_{} = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    custom_transformers.LowpassFilter(radius=0.{}),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])'''.format(i, i))
    exec('''test_cifar10_lowpass_{} = transforms.Compose([
    custom_transformers.LowpassFilter(radius=0.{}),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])'''.format(i, i))

for i in range(1, 10):
    exec('''train_ilsvrc_transform_lowpass_0_{} = transforms.Compose([
    transforms.RandomSizedCrop(224),
    transforms.RandomHorizontalFlip(),
    custom_transformers.LowpassFilter(radius=0.{}),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])'''.format(i, i))
    exec('''test_ilsvrc_transform_lowpass_0_{} = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    custom_transformers.LowpassFilter(radius=0.{}),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])'''.format(i, i))

for i in range(1, 10):
    exec('''train_ilsvrc_transform_lowpass_0_0_{} = transforms.Compose([
    transforms.RandomSizedCrop(224),
    transforms.RandomHorizontalFlip(),
    custom_transformers.LowpassFilter(radius=0.0{}),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])'''.format(i, i))
    exec('''test_ilsvrc_transform_lowpass_0_0_{} = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    custom_transformers.LowpassFilter(radius=0.0{}),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])'''.format(i, i))
