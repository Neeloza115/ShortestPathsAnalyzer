# Shortest Path Algorithms Project

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Project Structure](#project-structure)  
3. [Installation Requirements](#installation-requirements)  
4. [How to Run the Code](#how-to-run-the-code)  
5. [Experiments and Analysis](#experiments-and-analysis)  
6. [UML Design](#uml-design)  
7. [Team Charter](#team-charter)  
8. [Appendix: Code Navigation](#appendix-code-navigation)  

---

## Project Overview
This project explores shortest path algorithms using Python. The goal is to implement, analyze, and compare several algorithms, both on synthetic and real-world data, with a focus on correctness, efficiency, and practical applications.

**Algorithms implemented:**
- **Dijkstra's Algorithm (with k-relaxation)**  
- **Bellman-Ford Algorithm (with k-relaxation)**  
- **All-Pairs Shortest Path**  
- **A\* Algorithm**

**Real-world dataset:** London Subway Network, with stations as nodes and connections as edges. Edge weights are computed using geographical distance between stations.
---

## Installation Requirements
Python 3.8+ is required. Install the following packages:

```bash
pip install numpy pandas matplotlib networkx plotly
