import torch

def evaluate(model, loader, device):
    with torch.no_grad():
        model.eval()
        total = 0
        correct = 0

        for texts, targets, lengths in loader:
            texts = texts.to(device)
            targets = targets.to(device)
            lengths = lengths.to('cpu') #cpu로 맞춰줘야 pack_padded_seq 함수 사용 가능
            output = model(texts, lengths)

            output_index = torch.argmax(output, dim=1)
            total += targets.shape[0]
            correct += (targets==output_index).sum().item()
    
    acc = correct/total*100
    model.train()

    return acc
