Harsh Shah  Matthew Chang  Meetansh Gupta  Tanay Patankar


## Motivation

This project aims to help job seekers, whether they are new graduates, or have a few years of experience under their belt, to find jobs suited to their skill set. It is tough to find a job that accurately matches the skills that you have. Our project aims to make this easier by recommending jobs that best suit your skills. We also go one step further and suggest some skills that you can pick up in order to be a better match for jobs in your domain. We hope that with this application, we can increase the chances of a jobseeker getting a job that suits them.

## Implementation
We used a [dataset](https://www.kaggle.com/datasets/asaniczka/1-3m-linkedin-jobs-and-skills-2024) of 1.3 million job titles and skills scraped from LinkedIn. From this, we processed the job titles and the skills each one needed.

Then, we acquired BERT embeddings of each job's skills. The skills were formatted into a CSV and was used to get the embeddings. While it is known that GPT embeddings would be better for representing the data, it is slow and more expensive to use. Given the scope of our project, we decided to use BERT embeddings.

With these embeddings available with us now, we can now recomend jobs. With a given set of skills, we first find the BERT embeddings of these skills. We find the similarity score of this embedding with our database and rank the results. We utilize cosine similarity as our function. We return the top 10 of these results.

In order to recommend skills for upskilling, we go through the top 10000 most similar jobs and collect raw frequency counts of each skill. We recommend the top 10 most frequent skills which the user does not possess.

## Results

1. Result for query "python, linux, c++"
2. something for nursing? like stamina, compassion, patience, memory
3. idk some managerial position?

## Analysis
While recommending skills for upskilling, we realized that different ranges of top-k results gave different type of results. With k=100, we got more technical skills recommended to us. However, with a larger value of k=10000, we got skills that were more general and could be applied to a variety of jobs.

We also saw that TF-IDF gave a similar ranking as raw frequency counts and decided to use the less computationally intensive option.
