"""Create the sample document repository (Markdown, DOCX, TXT, PDF) in ./sample_docs."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tests.pdf_utils import make_pdf  # noqa: E402

DEFAULT_OUT = ROOT / "sample_docs"

ML_MD = """# Introduction to Machine Learning

Machine learning is a branch of artificial intelligence in which computers learn patterns from data instead of following hand-written rules. A model is trained on examples, and the trained model is then used to make predictions on new, unseen data.

## Types of machine learning

**Supervised learning** uses labeled examples. Each training example has an input and the correct output, for instance an email and the label "spam" or "not spam". Common supervised tasks are classification, which predicts a category, and regression, which predicts a number such as a house price.

**Unsupervised learning** works with unlabeled data. The goal is to discover structure, such as grouping similar customers together with clustering or reducing the number of features with dimensionality reduction.

**Reinforcement learning** trains an agent that takes actions in an environment and receives rewards or penalties. Over time the agent learns a policy that maximizes the total reward. Game-playing systems and robot controllers often use this approach.

## Training, validation and test data

A dataset is usually split into three parts. The training set, commonly about 70 percent of the data, is used to fit the model. The validation set, about 15 percent, is used to tune hyperparameters and to choose between models. The test set, about 15 percent, is held back until the end to give an honest estimate of how the model will perform on new data.

## Overfitting and underfitting

Overfitting happens when a model memorizes the training data, including its noise, and therefore performs poorly on new data. Underfitting happens when a model is too simple to capture the real pattern. Techniques that reduce overfitting include collecting more data, using regularization, applying dropout in neural networks, and stopping training early when the validation error starts to rise.

## Evaluation metrics

For classification, accuracy is the share of correct predictions. Precision is the share of predicted positives that are truly positive, and recall is the share of actual positives that the model finds. The F1 score is the harmonic mean of precision and recall. For regression, common metrics are mean absolute error (MAE) and root mean squared error (RMSE).
"""

RAG_MD = """# Retrieval-Augmented Generation (RAG) Overview

Retrieval-Augmented Generation, or RAG, combines a search system with a language model. Instead of relying only on what the model memorized during training, a RAG system first retrieves relevant passages from a document collection and then asks the model to answer using those passages.

## Why use RAG

Language models can produce fluent but wrong statements, which are called hallucinations. RAG reduces hallucinations because the answer is grounded in retrieved text, and the system can show the sources it used. RAG also lets a model answer questions about private or recent documents without retraining it.

## Main components

**Chunking.** Documents are split into short passages called chunks. Neighbouring chunks usually overlap by a small amount so that a fact at a boundary is not lost.

**Embeddings.** An embedding model converts each chunk into a vector of numbers. Passages with similar meaning have vectors that are close together. The same model converts the user's question into a vector.

**Vector database.** A vector database such as FAISS stores the chunk vectors and finds the nearest ones to a query vector quickly, usually with cosine similarity.

**Generator.** A language model receives the question and the retrieved chunks and writes the answer. A good prompt tells the model to use only the supplied context and to say when the answer is missing.

## Limitations

If retrieval misses the right passage, the generator cannot answer correctly. Small models may misread long contexts, and scanned documents need optical character recognition before they can be indexed.
"""

POLICY_PARAGRAPHS = [
    ("Title", "Northwind Analytics Remote Work Policy"),
    ("Heading 1", "Purpose"),
    ("Normal", "This policy explains how employees of Northwind Analytics may work from home while keeping teams connected and company data secure."),
    ("Heading 1", "Eligibility"),
    ("Normal", "Employees become eligible for remote or hybrid work after completing the 90-day probation period. Roles that need physical access to the laboratory, such as laboratory technicians, are not eligible for remote work."),
    ("Heading 1", "Work arrangements"),
    ("Normal", "Most employees follow the hybrid model and work in the office at least 2 days per week. Tuesday and Thursday are the anchor days when every team member is expected in the office. Fully remote work needs written approval from a vice president."),
    ("Heading 1", "Core hours"),
    ("Normal", "Everyone must be available between 10:00 and 15:00 in their local time zone. Outside core hours employees may arrange their schedule with their manager."),
    ("Heading 1", "Equipment and expenses"),
    ("Normal", "The company provides a laptop to every employee. Remote employees receive a one-time home-office stipend of $500 and a monthly internet reimbursement of $40. Equipment must be returned within 14 days after an employee leaves the company."),
    ("Heading 1", "Security requirements"),
    ("Normal", "Employees must use the company VPN whenever they access internal systems. Public Wi-Fi may only be used together with the VPN. A lost or stolen device must be reported to IT security within 24 hours."),
    ("Heading 1", "Approval process"),
    ("Normal", "Requests are submitted through the HR portal. The manager decides within 5 business days and HR confirms the arrangement in writing."),
]
POLICY_TABLE = [("Expense", "Amount", "Frequency"), ("Home-office stipend", "$500", "One time"),
                ("Internet reimbursement", "$40", "Monthly"), ("Co-working space", "Not reimbursed", "Not applicable")]

ENERGY_TXT = """Renewable Energy Overview

Renewable energy comes from sources that are naturally replenished, such as sunlight, wind, flowing water and heat from the earth. Using these sources produces far fewer greenhouse gas emissions than burning coal, oil or natural gas.

Solar power. Photovoltaic panels convert sunlight directly into electricity using semiconductor cells. Commercial panels typically reach an efficiency of 18 to 22 percent. Because the sun does not shine at night and is weaker in cloudy weather, a solar plant usually has a capacity factor of only 15 to 25 percent.

Wind power. Wind turbines turn the kinetic energy of moving air into electricity. Onshore wind farms usually have a capacity factor of 30 to 40 percent, while offshore wind farms reach 40 to 50 percent because winds at sea are stronger and steadier. Offshore projects are more expensive to build and maintain.

Hydropower. Hydropower is the largest source of renewable electricity worldwide. Dams store water in a reservoir and release it through turbines. Run-of-river plants use the natural flow without a large reservoir. Pumped storage plants pump water uphill when electricity is cheap and release it when demand is high, which makes them a form of energy storage.

Geothermal and biomass. Geothermal plants use steam or hot water from underground to drive turbines and can run around the clock. Biomass plants burn organic material such as wood waste or crop residue, or convert it into biofuels.

Challenges. The main challenge of wind and solar power is intermittency: output changes with the weather and the time of day. Grid operators handle this with energy storage, such as lithium-ion batteries and pumped hydro, with long-distance transmission lines, and with demand management. Integrating large shares of variable generation also requires upgrades to the electricity grid.

Costs. The levelized cost of energy, abbreviated LCOE, is the average cost per unit of electricity over the lifetime of a power plant, including construction, fuel and maintenance. The LCOE of solar and onshore wind has fallen steadily as technology improved and manufacturing scaled up.
"""

PHOTO_PAGES = [
    """Photosynthesis and Plant Biology

Photosynthesis is the process by which plants, algae and some bacteria convert light energy into chemical energy stored in sugars. The overall reaction combines carbon dioxide and water, in the presence of light, to produce glucose and oxygen. In plants, photosynthesis takes place inside organelles called chloroplasts.

Light-dependent reactions. The light-dependent reactions occur in the thylakoid membranes of the chloroplast. Chlorophyll absorbs mostly red and blue light and reflects green light, which is why leaves look green. The absorbed energy splits water molecules, releasing oxygen as a by-product. The reactions produce the energy carriers ATP and NADPH.""",
    """The Calvin cycle

The Calvin cycle is the second stage of photosynthesis. It takes place in the stroma, the fluid-filled space around the thylakoids inside the chloroplast. The cycle does not need light directly, but it depends on the ATP and NADPH made by the light-dependent reactions.

The Calvin cycle has three stages. In carbon fixation, the enzyme RuBisCO attaches carbon dioxide to a five-carbon sugar called RuBP. In reduction, ATP and NADPH are used to convert the resulting molecules into G3P, a three-carbon sugar. In regeneration, some of the G3P is used to rebuild RuBP so the cycle can continue, while the rest is used to make glucose and other carbohydrates.""",
    """Factors that affect the rate of photosynthesis

Several factors limit how fast photosynthesis can run. Light intensity matters because more light drives the light-dependent reactions faster, until the pigments are saturated. Carbon dioxide concentration matters because RuBisCO needs carbon dioxide for carbon fixation. Temperature matters because the enzymes involved work best at moderate temperatures; above about 40 degrees Celsius many enzymes begin to denature and the rate falls sharply. Water availability matters because plants close their stomata when water is scarce, which also reduces carbon dioxide intake.

The factor that is in shortest supply at a given moment is called the limiting factor. Increasing a factor that is not limiting will not raise the rate of photosynthesis.""",
]


def main(out: Path | None = None) -> Path:
    OUT = Path(out) if out else DEFAULT_OUT
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "01_intro_to_machine_learning.md").write_text(ML_MD, encoding="utf-8")
    (OUT / "05_rag_overview.md").write_text(RAG_MD, encoding="utf-8")
    (OUT / "03_renewable_energy_overview.txt").write_text(ENERGY_TXT, encoding="utf-8")
    make_pdf(PHOTO_PAGES, OUT / "04_photosynthesis_and_plant_biology.pdf")

    import docx

    document = docx.Document()
    for style, text in POLICY_PARAGRAPHS:
        document.add_paragraph(text, style=style)
    table = document.add_table(rows=len(POLICY_TABLE), cols=3)
    table.style = "Table Grid"
    for r, row in enumerate(POLICY_TABLE):
        for col, value in enumerate(row):
            table.cell(r, col).text = value
    document.save(OUT / "02_remote_work_policy.docx")
    print("Sample documents written to", OUT)
    return OUT


if __name__ == "__main__":
    main()
