# Zachary's Karate Club: Graph Neural Network

This is a lightweight, self-contained demonstration of the power of Graph Neural Networks (GNNs), applied to the famous Zachary's Karate Club sociological dataset.

## 🥋 The Problem
In the 1970s, anthropologist Wayne Zachary studied a university karate club. During the study, a conflict between the Instructor (Mr. Hi, Node 0) and the Administrator (Officer, Node 33) caused the club to split into two factions. 

Our goal is to **predict which faction each member joined** based *only* on the network of friendships within the club.

## ✨ The GNN Magic (Semi-Supervised Learning)
We are using a **Graph Convolutional Network (GCN)** written from scratch in PyTorch. 

The truly impressive part is that we train the network using **semi-supervised learning**. 
We only provide the labels for TWO nodes during training:
*   Node 0 (Instructor)
*   Node 33 (Administrator)

The GNN dynamically figures out the faction of the remaining 32 members purely by passing messages along the friendship edges in the graph!

## 🚀 How to Run in Spyder

1.  Open **Spyder**.
2.  Set your **Working Directory** to:
    `C:\Users\patha\Documents\GitHub\ML-and-DL\Graph_NN_samples\karate_club_gnn`
3.  Open `main.py` and hit **F5** (Run).
4.  Watch the console to see the accuracy climb from random to near-perfect in just a few epochs.
5.  Check the `plots/karate_club_gnn_results.png` file (or Spyder's Plot pane) for a beautiful side-by-side visualization of the graph and the GNN's 2D latent space embeddings!
