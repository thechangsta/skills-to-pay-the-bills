Harsh Shah,  Matthew Chang,  Meetansh Gupta,  Tanay Patankar

Link to our video: [here](https://youtu.be/mjT-HmlTLNA)

Link to our GitHub Repository: [here](https://github.com/thechangsta/skills-to-pay-the-bills)

## Motivation

This project aims to help job seekers, whether they are new graduates, or have a few years of experience under their belt, to find jobs suited to their skill set. It is tough to find a job that accurately matches the skills that you have. Our project aims to make this easier by recommending jobs that best suit your skills. We also go one step further and suggest some skills that you can pick up in order to be a better match for jobs in your domain. We hope that with this application, we can increase the chances of a jobseeker getting a job that suits them.

## Implementation

![implementation](https://github.com/thechangsta/skills-to-pay-the-bills/assets/65629932/e3e579da-139c-436e-b134-b9634bd79f83)

We used a [dataset](https://www.kaggle.com/datasets/asaniczka/1-3m-linkedin-jobs-and-skills-2024) of 1.3 million job titles and skills scraped from LinkedIn. From this, we processed the job titles and the skills each one needed.

Then, we acquired BERT embeddings of each job's skills. The skills were formatted into a CSV and was used to get the embeddings. While it is known that GPT embeddings would be better for representing the data, it is slow and more expensive to use. Given the scope of our project, we decided to use BERT embeddings.

With these embeddings available with us now, we can now recomend jobs. With a given set of skills, we first find the BERT embeddings of these skills. We find the similarity score of this embedding with our database and rank the results. We utilize cosine similarity as our function. We return the top 10 of these results.

In order to recommend skills for upskilling, we go through the top 10000 most similar jobs and collect raw frequency counts of each skill. We recommend the top 10 most frequent skills which the user does not possess.

## Results

1. Query: python, linux, management
![image](https://github.com/thechangsta/skills-to-pay-the-bills/assets/73650782/81a1d36a-960b-4a36-b13d-e4d6f53cbac1)

2. Query: content writing, graphic designing, adobe photoshop
![image](https://github.com/thechangsta/skills-to-pay-the-bills/assets/73650782/e9ddd0fa-51ae-4d9f-be54-58aa91cc7e53)

3. Query: CPR, intubation, cancer detection, surgery, appendectomy
![image](https://github.com/thechangsta/skills-to-pay-the-bills/assets/73650782/c819bdf4-9dca-4f03-b48f-0f822181722d)


## Analysis
While recommending skills for upskilling, we realized that different ranges of top-k results gave different type of results. With k=100, we got more technical skills recommended to us. However, with a larger value of k=10000, we got skills that were more general and could be applied to a variety of jobs.

We also saw that TF-IDF gave a similar ranking as raw frequency counts and decided to use the less computationally intensive option.
