# -*- coding: utf-8 -*-
"""
This script helps to:
1. Find the IP address of the Hue Bridge in the local network.
2. Create an authorized 'Application Key' IF one does not already exist.
3. List the v2 IDs of all connected lights and groups.
"""

import requests
import json
import ssl

# Disable SSL warnings for local, self-signed certificates of the Bridge
from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)


def discover_bridge_ip():
    """Tries to find the Bridge IP via Hue's discovery service."""
    print("Trying to find the Hue Bridge in the network...")
    try:
        response = requests.get('https://discovery.meethue.com/')
        bridges = response.json()
        if bridges:
            bridge_ip = bridges[0]['internalipaddress']
            print(f"Success! Bridge found at the IP address: {bridge_ip}")
            return bridge_ip
        else:
            print("No bridge found via the discovery service.")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error during automatic search: {e}")
        return None

def create_application_key(bridge_ip):
    """Creates a new Application Key on the Bridge."""
    url = f"https://{bridge_ip}/api"
    body = {"devicetype": "cat_detector_app#python_script", "generateclientkey": True}
    
    print("\n--- Step 1: Create Application Key ---")
    input("PLEASE PRESS THE LINK BUTTON ON YOUR HUE BRIDGE NOW and then press Enter here...")

    try:
        response = requests.post(url, json=body, verify=False, timeout=15)
        response_data = response.json()[0]
        
        if 'success' in response_data:
            app_key = response_data['success']['username']
            print("\nSUCCESS! An Application Key has been created.")
            print(f"Your Application Key is: {app_key}")
            return app_key
        elif 'error' in response_data:
            error_desc = response_data['error']['description']
            print(f"\nERROR: {error_desc}")
            if response_data['error']['type'] == 101:
                print("The link button was apparently not pressed in time.")
            return None
    except requests.exceptions.RequestException as e:
        print(f"An error has occurred: {e}")
        return None

def get_all_resources(bridge_ip, app_key):
    """Retrieves all controllable resources (lights, groups, etc.) from the Bridge."""
    if not app_key:
        return
        
    print("\n--- Step 2: Retrieve lights and groups ---")
    
    # Get lights
    url_lights = f"https://{bridge_ip}/clip/v2/resource/light"
    headers = {'hue-application-key': app_key}
    
    try:
        response = requests.get(url_lights, headers=headers, verify=False)
        response.raise_for_status()
        data = response.json()
        
        if 'data' in data and data['data']:
            print("\nFOUND LIGHTS:")
            for light in data['data']:
                light_id = light['id']
                light_name = light.get('metadata', {}).get('name', 'Unknown Name')
                print(f" - Name: '{light_name}', ID: {light_id}")
        else:
            print("\nNo individual lights found.")
            
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while retrieving the lights: {e}")

    # Get groups
    url_groups = f"https://{bridge_ip}/clip/v2/resource/grouped_light"
    try:
        response = requests.get(url_groups, headers=headers, verify=False)
        response.raise_for_status()
        data = response.json()

        if 'data' in data and data['data']:
            print("\nFOUND GROUPS (Rooms, Zones):")
            for group in data['data']:
                group_id = group['id']
                group_name = group.get('metadata', {}).get('name', 'Unknown Name')
                print(f" - Name: '{group_name}', ID: {group_id}")
            print("\nTIP: Using a group ID is more efficient than addressing individual lights!")
        else:
            print("\nNo groups found.")

    except requests.exceptions.RequestException as e:
        print(f"An error occurred while retrieving the groups: {e}")


if __name__ == "__main__":
    print("="*50)
    print(" Philips Hue Setup for the Cat Detector")
    print("="*50)

    # Determine IP address
    bridge_ip = discover_bridge_ip()
    if not bridge_ip:
        bridge_ip = input("Please enter the IP address of your Hue Bridge manually: ")

    # Check if a key already exists
    app_key = None
    has_key = input("Do you already have an Application Key (e.g., from the main script configuration)? (y/n): ").lower()
    
    if has_key == 'y':
        app_key = input("Please enter your existing Application Key: ")
    else:
        app_key = create_application_key(bridge_ip)

    # Get resources
    if app_key:
        get_all_resources(bridge_ip, app_key)
        
    print("\n--- Setup complete ---")
    print("Please copy the correct v2 IDs into the HUE_LIGHT_IDS list in your main script.")