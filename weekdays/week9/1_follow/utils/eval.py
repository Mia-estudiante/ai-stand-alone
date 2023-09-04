import torch

def evaluate(model, loader, device):
    with torch.no_grad():
        model.eval()
        total = 0
        correct = 0

        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            output = model(images)

            output_index = torch.argmax(output, dim=1)
            total += targets.shape[0]
            correct += (targets==output_index).sum().item()
    
    acc = correct/total*100
    model.train()

    return acc

def evaluate_class(model, loader, device, num_classes):
    with torch.no_grad():
        model.eval()
        total = torch.zeros(num_classes) 
        correct = torch.zeros(num_classes) 

        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            output = model(images)

            output_index = torch.argmax(output, dim=1)
            
            for _class in range(num_classes):
                total[_class] += (targets==_class).sum().item()
                correct[_class] += ((targets==_class) * (output_index==_class)).sum().item()
    
    acc_class = correct/total*100
    model.train()
    return acc_class
