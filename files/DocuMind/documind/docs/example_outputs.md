# Example outputs

Generated on the sample documents. Mode: **OFFLINE FALLBACK (hashing embedder + extractive answers), not the real models**.

> These outputs come from the dependency-free fallback so they could be produced without downloading models.
> With FLAN-T5 and MiniLM (the default in Colab) answers are more fluent and retrieval handles paraphrased questions.

## Q: How much is the home-office stipend?

**Answer (answer):** Remote employees receive a one-time home-office stipend of $500 and a monthly internet reimbursement of $40. Expense: Home-office stipend; Amount: $500; Frequency: One time.

- [1] `02_remote_work_policy.docx`, page n/a, similarity 0.34 (cited): Most employees follow the hybrid model and work in the office at least 2 days per week. Tuesday and Thursday are the anchor days when every team member is expec...
- [2] `02_remote_work_policy.docx`, page n/a, similarity 0.29 (retrieved, not cited): Northwind Analytics Remote Work Policy Purpose This policy explains how employees of Northwind Analytics may work from home while keeping teams connected and co...
- [3] `02_remote_work_policy.docx`, page n/a, similarity 0.20 (cited): Equipment must be returned within 14 days after an employee leaves the company. Security requirements Employees must use the company VPN whenever they access in...

## Q: What is the difference between supervised and unsupervised learning?

**Answer (answer):** Unsupervised learning works with unlabeled data. Types of machine learning Supervised learning uses labeled examples.

- [1] `01_intro_to_machine_learning.md`, page n/a, similarity 0.27 (cited): Common supervised tasks are classification, which predicts a category, and regression, which predicts a number such as a house price. Unsupervised learning work...
- [2] `01_intro_to_machine_learning.md`, page n/a, similarity 0.17 (cited): Introduction to Machine Learning Machine learning is a branch of artificial intelligence in which computers learn patterns from data instead of following hand-w...

## Q: Where does the Calvin cycle take place?

**Answer (answer):** It takes place in the stroma, the fluid-filled space around the thylakoids inside the chloroplast. In plants, photosynthesis takes place inside organelles called chloroplasts.

- [1] `04_photosynthesis_and_plant_biology.pdf`, page 2, similarity 0.44 (cited): The Calvin cycle The Calvin cycle is the second stage of photosynthesis. It takes place in the stroma, the fluid-filled space around the thylakoids inside the c...
- [2] `04_photosynthesis_and_plant_biology.pdf`, page 2, similarity 0.35 (retrieved, not cited): In regeneration, some of the G3P is used to rebuild RuBP so the cycle can continue, while the rest is used to make glucose and other carbohydrates....
- [3] `04_photosynthesis_and_plant_biology.pdf`, page 1, similarity 0.13 (cited): Photosynthesis and Plant Biology Photosynthesis is the process by which plants, algae and some bacteria convert light energy into chemical energy stored in suga...

## Q: What is the capacity factor of offshore wind farms?

**Answer (answer):** Onshore wind farms usually have a capacity factor of 30 to 40 percent, while offshore wind farms reach 40 to 50 percent because winds at sea are stronger and steadier.

- [1] `03_renewable_energy_overview.txt`, page n/a, similarity 0.42 (cited): Wind power. Wind turbines turn the kinetic energy of moving air into electricity. Onshore wind farms usually have a capacity factor of 30 to 40 percent, while o...
- [2] `03_renewable_energy_overview.txt`, page n/a, similarity 0.24 (retrieved, not cited): Renewable Energy Overview Renewable energy comes from sources that are naturally replenished, such as sunlight, wind, flowing water and heat from the earth. Usi...
- [3] `04_photosynthesis_and_plant_biology.pdf`, page 3, similarity 0.15 (retrieved, not cited): Water availability matters because plants close their stomata when water is scarce, which also reduces carbon dioxide intake. The factor that is in shortest sup...

## Q: Why does RAG reduce hallucinations?

**Answer (answer):** Why use RAG Language models can produce fluent but wrong statements, which are called hallucinations. RAG reduces hallucinations because the answer is grounded in retrieved text, and the system can show the sources it used.

- [1] `05_rag_overview.md`, page n/a, similarity 0.35 (cited): Retrieval-Augmented Generation (RAG) Overview Retrieval-Augmented Generation, or RAG, combines a search system with a language model. Instead of relying only on...
- [2] `01_intro_to_machine_learning.md`, page n/a, similarity 0.12 (retrieved, not cited): Overfitting and underfitting Overfitting happens when a model memorizes the training data, including its noise, and therefore performs poorly on new data. Under...
- [3] `03_renewable_energy_overview.txt`, page n/a, similarity 0.13 (retrieved, not cited): Run-of-river plants use the natural flow without a large reservoir. Pumped storage plants pump water uphill when electricity is cheap and release it when demand...

## Q: What about the internet?

**Answer (answer):** Remote employees receive a one-time home-office stipend of $500 and a monthly internet reimbursement of $40. Expense: Home-office stipend; Amount: $500; Frequency: One time.

- [1] `02_remote_work_policy.docx`, page n/a, similarity 0.26 (cited): Most employees follow the hybrid model and work in the office at least 2 days per week. Tuesday and Thursday are the anchor days when every team member is expec...
- [2] `02_remote_work_policy.docx`, page n/a, similarity 0.24 (cited): Equipment must be returned within 14 days after an employee leaves the company. Security requirements Employees must use the company VPN whenever they access in...
- [3] `02_remote_work_policy.docx`, page n/a, similarity 0.22 (retrieved, not cited): Northwind Analytics Remote Work Policy Purpose This policy explains how employees of Northwind Analytics may work from home while keeping teams connected and co...

## Q: Who won the 2018 football world cup?

**Answer (not_found):** I couldn't find this information in the uploaded documents.

