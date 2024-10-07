from transformers import pipeline
from PIL import Image

# Load the image and convert to RGBA to avoid the transparency issue
image_path = "/Users/glenn/dochandel/1131w-oGcvrLBYpGk.webp"
image = Image.open(image_path).convert("RGBA")
image.save("/Users/glenn/dochandel/invoice_rgba.png")  # Save the converted image

# Load the converted image in the pipeline
nlp = pipeline(
    "document-question-answering",
    model="impira/layoutlm-document-qa",
)

# Get answers to each question and store the result
bank_name= nlp("/Users/glenn/dochandel/invoice_rgba.png", "What is the nsme of the bank?")
result_purchase_amount = nlp("/Users/glenn/dochandel/invoice_rgba.png", "What is the total on the bill")
result_billed_to = nlp("/Users/glenn/dochandel/invoice_rgba.png", "Who is the invoice billed to")
mail = nlp("/Users/glenn/dochandel/invoice_rgba.png", "what is the email of billed to custommer ")

# Print the results
print("What is the name of the bankr:", bank_name[0])
print("Purchase Amount:", result_purchase_amount[0])
print("Billed to:", result_billed_to[0])
print("Email is:", mail[0])
