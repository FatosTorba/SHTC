from tqdm import tqdm
import torch
from constrained_module.constrain_utils import constrained_loss_dfs, constrained_loss_bfs


def train_model(model, train_loader, val_loader, num_epochs=10, next_token_mapping=None, set_all_labels=None,
                file_path_save="model.pt", is_constrained=True):
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    loss_f = torch.nn.NLLLoss(reduction="mean")
    best_val_loss = float('inf')
    validation_step = int(len(train_loader.dataset) / (3 * train_loader.batch_size)) + 1
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        step = -1
        for inputs, targets, targets_multi_hot in tqdm(train_loader):
            step += 1
            inputs, targets = list(inputs), list(targets)
            optimizer.zero_grad()
            if is_constrained:
                labels, logits = model(inputs, targets)
                if model.target_type == "dfs":
                    loss = constrained_loss_dfs(model, loss_f, logits, labels,
                                                next_token_mapping, model.tokenizer)
                elif model.target_type == "bfs":
                    loss = constrained_loss_bfs(model, loss_f, logits, labels,
                                                next_token_mapping, set_all_labels, model.tokenizer)
            else:
                labels, loss = model(inputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if step % validation_step == 0 and epoch > 0:
                model.eval()
                total_val_loss = 0.0
                with torch.no_grad():
                    for inputs, targets, targets_multi_hot in tqdm(val_loader):
                        inputs, targets = list(inputs), list(targets)
                        optimizer.zero_grad()
                        if is_constrained:
                            labels, logits = model(inputs, targets)
                            if model.target_type == "dfs":
                                loss = constrained_loss_dfs(model, loss_f, logits, labels,
                                                            next_token_mapping, model.tokenizer)
                            elif model.target_type == "bfs":
                                loss = constrained_loss_bfs(model, loss_f, logits, labels,
                                                            next_token_mapping, set_all_labels,
                                                            model.tokenizer)
                        else:
                            labels, loss = model(inputs, targets)

                        total_val_loss += loss.item()
                avg_val_loss = total_val_loss / len(val_loader)
                # Save the best model based on validation loss
                if avg_val_loss < best_val_loss and epoch > 1:
                    best_val_loss = avg_val_loss
                    torch.save(model, file_path_save)
                print(f"Epoch [{epoch + 1}/{num_epochs}] - Avg Validation Loss: {avg_val_loss:.6f}")
        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Avg Train Loss: {avg_train_loss:.6f}")
