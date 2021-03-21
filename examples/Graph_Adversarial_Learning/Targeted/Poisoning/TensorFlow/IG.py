import graphgallery as gg
from graphgallery import functional as gf
from graphgallery.datasets import NPZDataset

data = NPZDataset('cora',
                  root="~/GraphData/datasets/",
                  verbose=False,
                  transform="standardize")

graph = data.graph
splits = data.split_nodes(random_state=15)

################### Surrogate model ############################
trainer = gg.gallery.nodeclas.DenseGCN(seed=123).make_data(graph).build()
his = trainer.fit(splits.train_nodes,
                  splits.val_nodes,
                  verbose=1,
                  epochs=100)

################### Attacker model ############################
target = 1
attacker = gg.attack.targeted.IG(graph, seed=123).process(trainer)
attacker.attack(target)

################### Victim model ############################
# Before attack
trainer = gg.gallery.nodeclas.GCN(seed=123).make_data(graph).build()
his = trainer.fit(splits.train_nodes,
                  splits.val_nodes,
                  verbose=1,
                  epochs=100)
original_predict = trainer.predict(target, transform="softmax")

# After attack
trainer = gg.gallery.nodeclas.GCN(seed=123).make_data(attacker.g).build()
his = trainer.fit(splits.train_nodes,
                  splits.val_nodes,
                  verbose=1,
                  epochs=100)
perturbed_predict = trainer.predict(target, transform="softmax")

################### Results ############################
print("original prediction", original_predict)
print("perturbed prediction", perturbed_predict)
target_label = graph.node_label[target]
print(f"The True label of node {target} is {target_label}.")
print(
    f"The probability of prediction has gone down {original_predict[target_label]-perturbed_predict[target_label]}"
)
"""original prediction [0.00212943 0.0030072  0.90525377 0.03167017 0.01139321 0.00445553
 0.04209068]
perturbed prediction [0.03303587 0.71460384 0.02212887 0.04821165 0.02003909 0.07325915
 0.08872161]
The True label of node 1 is 2.
The probability of prediction has gone down 0.8831248879432678"""
