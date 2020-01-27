MyPoorlyDocumentedInfo = provider()

MyFooInfo = provider(
    doc = "Stores information about a foo.",
    fields = ["bar", "baz"],
)

MyVeryDocumentedInfo = provider(
    doc = """
A provider with some really neat documentation.
Look on my works, ye mighty, and despair!
""",
    fields = {
        "favorite_food": "A string representing my favorite food",
        "favorite_color": "A string representing my favorite color",
    },
)
