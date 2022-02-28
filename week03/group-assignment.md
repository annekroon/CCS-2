

# Group assignment: Build a recommender systems
### Deadline: Week 7, Friday 20--5.

Form groups of around 4 students (5 is the maximum). You will receive a dataset from your tutorial lecture. This dataset will form the basis of the assignment. Your task is to explore this dataset, describe it in a meaningful, data-scientific way, and ultimately, to build a recommender system.


  1.  **Explore**, **pre-process**, and **clean** the dataset.
      - Explore the dataset, and inspect what type of relevant variables are present, what data can be used. Select which variables might be of interest and can be used later on.
      - Feature engineering is an important step here (keeping in mind the type of descriptive analysis you want to conduct in step 2).
      - The literature and code examples from week 1 and week 2 should help you here.

  2.  **Describe** the dataset using an inductive analysis.
      - Provide a clear description of data you will be working with. E.g., describe the most interesting variables in terms of data `type`, number of unique observations, mean, distribution, etc.
        - Plotting the data, to visualise some of the relations in the dataset, is appreciated.
      - Describe the dataset using some of the techniques as discussed in week 3 and week 4. For example, apply LDA to describe the number of topics present in the dataset.

  3.  **Build a recommender system**
      - Build a recommender system, based on the insights from week 6. It's up to you to decide whether you build a knowledge-based or content-based recommender system.
      - Think about relevant features that you want to use in your algorithm design. Based on which features do you want to recommend content?

---

## The final assignment consists of the following:

1.  A paper, consisting of..

  1.  Method section
    - A description of the steps you took, which type of variables were selected and how they were transformed.
    - Explain your analytical strategy;
          - What techniques (and why) will you be using to describe the dataset?
          - What type of recommender system are you building? Why?  

  2.  Results section
    - A description of the dataset (how many observations, what type of variables)
    - Results of the inductive analysis (e.g., description of the topics you've found).
    - Demonstration of the recommender system; explanation of how it works, and some examples from the type of recommendations you get for different types of input.

2. The code belonging to the project.

  -  A set of scripts used to preprocess and analyse the data.
  - The output files (if you work in Jupyter Notebooks, code and output can be integrated).
  - Preferably, your code is well-documented and available in a github repo.
