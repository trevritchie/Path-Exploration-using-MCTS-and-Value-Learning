# **Path Exploration using MCTS & Value Learning

## **Objective**

Use MCTS to explore the path between Charleston, SC and Charlotte, NC, based on the list of cities provided in the previous assignment.

---

## **Assignment Details**

### **Shortest Path**

- Generate a graph/grid/tree data structure between the starting point Charleston, and the destination, Charlotte.
- Calculate the shortest path.
	1. Take the cities, not in the shortest path and add it to some form of list.
	2. From this list select 5 random locations to stop at - **Delivery Points**.
		- Once selected, these **Delivery points** will not change across different iterations of the next process

### **MCTS and Value Learning**

- Traverse your graph/grid/tree of cities using MCTS
	1. If the node reached, is part of shortest path, add 15 energy/value points.
	2. If not, lose 5 energy/value points.
	3. If you reach a **Delivery Points**, add 30 points.
	4. If you reach your destination, add 100 points and terminate traversal
	5. If the energy reaches a certain negative value(threshold), terminate traversal and lose 100 points.
		- you must experiment, and play around with this threshold value to allow enough exploration.
	6. Iterate this process to refine the path between Charleston and Charlotte while maximizing point/energy value.


### **Experiment**

- Tweak the reward values and stop traversal when you hit negative enery/value points.
	- Does this make the process faster?
	- Do you need to keep track of these events and set extra negative rewards for the agent when this occurs?
- What if there is not shortest path metric and you start with certain energy - lets say 50.
	- Does this lead to a better solution with higher final reward?
- Can you discard MCTS altogether and just use random traversal or other tree traversal order?
	- How should the reward system change then?
- Should the Agent only become aware of the rewards at the **Delivery Points** when it explores the parent node of these locations and finds these rewards through the Expansion/Rollout process of MCTS, or would it help to be aware of the these **Delivery Points** at all times? (Partially observed/Fully Observed).
---
## **Submission Requirements**

- One submission per group is enough.
- Submit `PathOpt-Learning.py`
- Submit `README.md` detailing your approach, decisions, design choices and results. This is an open ended assignment - Add params whenever necessary.
	- Follow proper README structure (add relevant headings and descriptions)
	- Add Group Members as authors

## **Grading Criteria (100 points)**

| Criteria                                             | Points |
| ---------------------------------------------------- | ------ |
| **Correct Implementation of MCTS for this scenario** | 30     |
| **Correct implementation of the learning process**   | 30     |
| **Path Exploration**                                 | 20     |
| **Appropriate console output showcasing results**    | 5      |
| **Documentation & Code Readability**                 | 15     |
Points will be calculated out of 100 and compressed down to 25 points.

---

### **Notes:**
- Ensure your final script runs without errors and produces meaningful results.
