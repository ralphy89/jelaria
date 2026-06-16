import string
import easyocr

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}




import csv
import requests


def init_csv(output_path):
    """
    Initialise le fichier CSV avec les colonnes.
    """
    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame_nmr",
            "car_id",
            "car_bbox",
            "license_plate_bbox",
            "license_plate_bbox_score",
            "license_number",
            "license_number_score",
            "status"
        ])


def append_csv(output_path, frame_nmr, car_id, car_bbox, license_plate_bbox,
               license_plate_bbox_score, license_number, license_number_score,
               status="PLATE_RECOGNIZED"):
    """
    Ajoute une ligne dans le CSV dès qu'une plaque valide est détectée.
    """
    with open(output_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            frame_nmr,
            car_id,
            car_bbox,
            license_plate_bbox,
            license_plate_bbox_score,
            license_number,
            license_number_score,
            status
        ])


def send_to_server(server_url, payload):
    """
    Envoie le résultat au serveur.
    Si le serveur n'est pas disponible, le programme continue.
    """
    try:
        response = requests.post(server_url, json=payload, timeout=3)

        if response.status_code in [200, 201]:
            print("Résultat envoyé au serveur")
            return True
        else:
            print(f"Erreur serveur: {response.status_code}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"Serveur non disponible: {e}")
        return False




def write_csv(results, output_path):
    """
    Write the results to a CSV file.

    Args:
        results (dict): Dictionary containing the results.
        output_path (str): Path to the output CSV file.
    """
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_bbox',
                                                'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                                                'license_number_score'))

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                print(results[frame_nmr][car_id])
                if 'car' in results[frame_nmr][car_id].keys() and \
                   'license_plate' in results[frame_nmr][car_id].keys() and \
                   'text' in results[frame_nmr][car_id]['license_plate'].keys():
                    f.write('{},{},{},{},{},{},{}\n'.format(frame_nmr,
                                                            car_id,
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['car']['bbox'][0],
                                                                results[frame_nmr][car_id]['car']['bbox'][1],
                                                                results[frame_nmr][car_id]['car']['bbox'][2],
                                                                results[frame_nmr][car_id]['car']['bbox'][3]),
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][0],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][1],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][2],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][3]),
                                                            results[frame_nmr][car_id]['license_plate']['bbox_score'],
                                                            results[frame_nmr][car_id]['license_plate']['text'],
                                                            results[frame_nmr][car_id]['license_plate']['text_score'])
                            )
        f.close()


import re

def license_complies_format(text):
    """
    Vérifie si le texte ressemble à une plaque lisible.
    Version flexible pour la phase prototype.
    """
    if text is None:
        return False

    text = text.upper().replace(" ", "").replace("-", "")

    # Accepte uniquement lettres et chiffres
    if not re.match(r"^[A-Z0-9]+$", text):
        return False

    # Longueur raisonnable pour une plaque
    if len(text) < 5 or len(text) > 10:
        return False

    return True


def format_license(text):
    """
    Nettoie simplement le texte OCR pour le prototype.
    """
    if text is None:
        return None

    text = text.upper()
    text = text.replace(" ", "")
    text = text.replace(".", "")
    text = text.replace(",", "")
    text = text.replace("_", "")
    
    return text


def read_license_plate(license_plate_crop):
    """
    Lit le texte d'une plaque à partir d'une image recadrée.
    Retourne le texte nettoyé et le score OCR.
    """

    detections = reader.readtext(license_plate_crop)

    best_text = None
    best_score = 0

    for detection in detections:
        bbox, text, score = detection

        text = format_license(text)

        if license_complies_format(text) and score > best_score:
            best_text = text
            best_score = score

    if best_text is not None:
        return best_text, best_score

    return None, None


def get_car(license_plate, vehicle_track_ids):
    """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates.

    Args:
        license_plate (tuple): Tuple containing the coordinates of the license plate (x1, y1, x2, y2, score, class_id).
        vehicle_track_ids (list): List of vehicle track IDs and their corresponding coordinates.

    Returns:
        tuple: Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID.
    """
    x1, y1, x2, y2, score, class_id = license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]

    return -1, -1, -1, -1, -1
