import itertools
import sys
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas as pdf_canvas
from vpython import canvas, sphere, cylinder, color, vector, label, rate, cos, sin

# Genetic probabilities
PROBS = {
    "gene": {2: 0.01, 1: 0.03, 0: 0.96},
    "trait": {
        2: {True: 0.65, False: 0.35},
        1: {True: 0.56, False: 0.44},
        0: {True: 0.01, False: 0.99}
    },
    "mutation": 0.01
}

def load_data(filename):
    """
    Load gene and trait data from a CSV file into a dictionary.
    """
    try:
        data = pd.read_csv(filename)
    except FileNotFoundError:
        sys.exit("Error: File not found. Please provide a valid CSV file.")

    people = data.set_index("name").to_dict("index")
    
    for person in people:
        people[person]["trait"] = (
            True if people[person]["trait"] == 1 else
            False if people[person]["trait"] == 0 else None
        )
    return people

def powerset(s):
    """
    Return all possible subsets of set `s`.
    """
    return [set(s) for s in itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(len(s) + 1)
    )]

def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute the joint probability for a specific configuration.
    """
    joint_prob = 1

    for person in people:
        genes = 2 if person in two_genes else 1 if person in one_gene else 0
        trait = person in have_trait

        mother = people[person]['mother']
        father = people[person]['father']

        if not mother or not father:
            prob = PROBS["gene"][genes]
        else:
            mother_prob = inherit_prob(mother, one_gene, two_genes)
            father_prob = inherit_prob(father, one_gene, two_genes)

            if genes == 2:
                prob = mother_prob * father_prob
            elif genes == 1:
                prob = mother_prob * (1 - father_prob) + (1 - mother_prob) * father_prob
            else:
                prob = (1 - mother_prob) * (1 - father_prob)

        prob *= PROBS["trait"][genes][trait]
        joint_prob *= prob

    return joint_prob

def inherit_prob(parent, one_gene, two_genes):
    """
    Calculate the probability of inheriting a gene from a parent.
    """
    if parent in two_genes:
        return 1 - PROBS["mutation"]
    elif parent in one_gene:
        return 0.5
    else:
        return PROBS["mutation"]

def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Update probabilities with the joint probability `p`.
    """
    for person in probabilities:
        genes = 2 if person in two_genes else 1 if person in one_gene else 0
        trait = person in have_trait

        probabilities[person]["gene"][genes] += p
        probabilities[person]["trait"][trait] += p

def normalize(probabilities):
    """
    Normalize probabilities so that each sum is 1.
    """
    for person in probabilities:
        gene_total = sum(probabilities[person]["gene"].values())
        trait_total = sum(probabilities[person]["trait"].values())

        for key in probabilities[person]["gene"]:
            probabilities[person]["gene"][key] /= gene_total

        for key in probabilities[person]["trait"]:
            probabilities[person]["trait"][key] /= trait_total

def generate_pdf_report(probabilities, filename="Genetic_Report.pdf"):
    """
    Generate a PDF report summarizing genetic analysis results.
    """
    c = pdf_canvas.Canvas(filename, pagesize=letter)
    c.setFont("Helvetica", 12)
    c.drawString(30, 750, "Patient Genetic Analysis Report")
    c.drawString(30, 735, "----------------------------------------")
    y = 720

    for person, data in probabilities.items():
        c.drawString(30, y, f"Patient: {person}")
        y -= 15

        gene_probs = data["gene"]
        max_gene = max(gene_probs, key=gene_probs.get)

        c.drawString(50, y, f"- Likely gene copies: {max_gene} ({gene_probs[max_gene]:.2%})")
        y -= 15

        trait_probs = data["trait"]
        has_trait = trait_probs[True] > trait_probs[False]
        c.drawString(50, y, f"- Trait analysis: {'Likely to have trait' if has_trait else 'Unlikely to have trait'}")
        y -= 30

    c.save()
    print(f"PDF report generated: {filename}")


def simulate_3d_dna():
    """
    Simulate a realistic 3D double-helix DNA model using VPython, 
    with a double strand, base-pair bonds, and realistic twisting.
    """
    # Set up the canvas
    scene = canvas(width=800, height=600, background=color.white)
    scene.center = vector(0, 0, 0)
    scene.append_to_caption("""
        <style>
            body {
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
            }
        </style>
    """)
    scene.camera.pos = vector(0, 5, 25)
    scene.camera.axis = vector(0, -5, -25)

    # DNA parameters
    helix_radius = 1       # Radius of the DNA helix
    helix_height = 10      # Total height of the helix
    num_turns = 10         # Number of helical turns
    base_pairs_per_turn = 10  # Number of base pairs per turn
    total_base_pairs = num_turns * base_pairs_per_turn  # Total base pairs
    spacing = helix_height / total_base_pairs  # Vertical spacing per base pair

    # Colors for the bases
    base_colors = [color.red, color.blue, color.green, color.yellow]
    vertical_offset = -helix_height / 2  # Center the helix vertically

    # Create the double helix
    for i in range(total_base_pairs):
        theta = (2 * 3.14159 * i) / base_pairs_per_turn  # Angle for helical twist

        # Left strand
        left_x = helix_radius * cos(theta)
        left_y = helix_radius * sin(theta)
        left_z = i * spacing + vertical_offset
        left_base = sphere(pos=vector(left_x, left_y, left_z), radius=0.1, color=base_colors[i % len(base_colors)])

        # Right strand
        right_x = -helix_radius * cos(theta)
        right_y = -helix_radius * sin(theta)
        right_z = i * spacing + vertical_offset
        right_base = sphere(pos=vector(right_x, right_y, right_z), radius=0.1, color=base_colors[(i + 2) % len(base_colors)])

        # Base-pair bond
        bond = cylinder(pos=vector(left_x, left_y, left_z),
                        axis=vector(right_x - left_x, right_y - left_y, right_z - left_z),
                        radius=0.05, color=color.white)

    # Animate the rotation for a better view
    angle = 0
    while True:
        rate(30)
        angle += 0.01
        scene.camera.pos = vector(20 * cos(angle), 10, 20 * sin(angle))
        scene.camera.axis = vector(-scene.camera.pos.x, -scene.camera.pos.y, -scene.camera.pos.z)

if __name__ == "__main__":
    simulate_3d_dna()











