"""
Collapse the MIDA head atlas's ~100 fine-grained tissue labels into six
coarse groups (white matter, gray matter, CSF, bone, scalp, muscle) used by
the FEM material model, and crop to the region of interest (ROI) used by the
rest of the pipeline.
"""

from pathlib import Path

import nibabel as nib
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent

# Load original NIfTI image
# In this file, Y is the vertical, Z is the horizontal, X is backward/forward
img = nib.load(SCRIPT_DIR / "MIDA_v1.nii")
data = img.get_fdata()
affine = img.affine

tissue_groups = {
    "white_matter": [
        (3, "Pineal Body"),
        (9, "Cerebellum White Matter"),
        (12, "Brain White Matter"),
        (18, "Optic Tract"),
        (19, "Hypophysis or Pituitary Gland"),
        (22, "Commissura (Anterior)"),
        (23, "Commissura (Posterior)"),
        (100, "Cerebral Peduncles"),
        (101, "Optic Chiasm"),
    ],
    "gray_matter": [
        (2, "Cerebellum Gray Matter"),
        (4, "Amygdala"),
        (5, "Hippocampus"),
        (7, "Caudate Nucleus"),
        (8, "Putamen"),
        (10, "Brain Gray Matter"),
        (11, "Brainstem Midbrain"),
        (13, "Spinal Cord"),
        (14, "Brainstem Pons"),
        (15, "Brainstem Medulla"),
        (16, "Nucleus Accumbens"),
        (17, "Globus Pallidus"),
        (20, "Mammillary Body"),
        (21, "Hypothalamus"),
        (99, "Substantia Nigra"),
        (116, "Thalamus"),
    ],
    "csf": [
        (1, "Dura"),
        (6, "CSF Ventricles"),
        (32, "CSF General"),
        (24, "Blood Arteries"),
        (25, "Blood Veins"),
    ],
    "bone": [
        (36, "Mandible"),
        (40, "Skull"),
        (41, "Teeth"),
        (44, "Vertebra - C1 (atlas)"),
        (45, "Vertebra - C2 (axis)"),
        (46, "Vertebra - C3"),
        (47, "Vertebra - C4"),
        (48, "Vertebra - C5"),
        (49, "Intervertebral Discs"),
        (52, "Skull Diploë"),
        (53, "Skull Inner Table"),
        (54, "Skull Outer Table"),
        (87, "Hyoid Bone"),
    ],
    "scalp": [
        (33, "Ear Cochlea"),
        (34, "Ear Semicircular Canals"),
        (35, "Ear Auricular Cartilage (Pinna)"),
        (37, "Mucosa"),
        (39, "Nasal Septum (Cartilage)"),
        (43, "Adipose Tissue"),
        (51, "Epidermis/Dermis"),
        (55, "Eye Lens"),
        (56, "Eye Retina/Choroid/Sclera"),
        (57, "Eye Vitreous"),
        (58, "Eye Cornea"),
        (59, "Eye Aqueous"),
        (61, "Tendon - Galea Aponeurotica"),
        (62, "Subcutaneous Adipose Tissue"),
        (85, "Ear Auditory Canal"),
        (86, "Ear Pharyngotympanic Tube"),
        (88, "Submandibular Gland"),
        (89, "Parotid Gland"),
        (90, "Sublingual Gland"),
        (98, "Tendon - Temporalis Tendon"),
    ],
    "muscle": [
        (38, "Muscle (General)"),
        (42, "Tongue"),
        (60, "Muscle - Platysma"),
        (63, "Muscle - Temporalis/Temporoparietalis"),
        (64, "Muscle - Occipitiofrontalis - Frontal Belly"),
        (65, "Muscle - Lateral Pterygoid"),
        (66, "Muscle - Masseter"),
        (67, "Muscle - Splenius Capitis"),
        (68, "Muscle - Sternocleidomastoid"),
        (69, "Muscle - Occipitiofrontalis - Occipital Belly"),
        (70, "Muscle - Trapezius"),
        (71, "Muscle - Mentalis"),
        (72, "Muscle - Depressor Anguli Oris"),
        (73, "Muscle - Depressor Labii"),
        (74, "Muscle - Nasalis"),
        (75, "Muscle - Orbicularis Oris"),
        (76, "Muscles - Procerus"),
        (77, "Muscle - Levator Labii Superioris"),
        (78, "Muscle - Zygomaticus Major"),
        (79, "Muscle - Orbicularis Oculi"),
        (80, "Muscle - Levator Scapulae"),
        (81, "Muscle - Medial Pterygoid"),
        (82, "Muscle - Zygomaticus Minor"),
        (83, "Muscles - Risorius"),
        (84, "Muscle - Buccinator"),
        (91, "Muscle - Superior Rectus"),
        (92, "Muscle - Medial Rectus"),
        (93, "Muscle - Lateral Rectus"),
        (94, "Muscle - Inferior Rectus"),
        (95, "Muscle - Superior Oblique"),
        (96, "Muscle - Inferior Oblique"),
        (102, "Cranial Nerve I - Olfactory"),
        (103, "Cranial Nerve II - Optic"),
        (104, "Cranial Nerve III - Oculomotor"),
        (105, "Cranial Nerve IV - Trochlear"),
        (106, "Cranial Nerve V - Trigeminal"),
        (107, "Cranial Nerve V2 - Maxillary Division"),
        (108, "Cranial Nerve V3 - Mandibular Division"),
        (109, "Cranial Nerve VI - Abducens"),
        (110, "Cranial Nerve VII - Facial"),
        (111, "Cranial Nerve VIII - Vestibulocochlear"),
        (112, "Cranial Nerve IX - Glossopharyngeal"),
        (113, "Cranial Nerve X - Vagus"),
        (114, "Cranial Nerve XI - Accessory"),
        (115, "Cranial Nerve XII - Hypoglossal"),
    ],
}


def tags_for(group_name):
    return [tag for tag, _ in tissue_groups.get(group_name, [])]


# Convert data to integer type if not already (to work with labels)
data = data.astype(np.uint16)

# Replace all voxels with specified label by 0 (air, background)
labels_air = [26, 27, 28, 29, 30, 31, 97]
labels_to_remove = [50]
data[np.isin(data, labels_to_remove)] = 0
data[np.isin(data, tags_for("white_matter"))] = 1000
data[np.isin(data, tags_for("gray_matter"))] = 2000
data[np.isin(data, tags_for("csf"))] = 3000
data[np.isin(data, tags_for("bone"))] = 4000
data[np.isin(data, tags_for("scalp"))] = 5000
data[np.isin(data, tags_for("muscle"))] = 6000
data[np.isin(data, labels_air)] = 7000
data = (data // 1000).astype(np.uint8)

# Crop to region of interest (half head)
roi = data[15:460, 160:470, 8:173]

roi_img = nib.Nifti1Image(roi, affine)

# Save new NIfTI image
nib.save(roi_img, SCRIPT_DIR / "MIDA_roi_uint8_group_full.nii")