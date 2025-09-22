import osmnx as ox
import networkx as nx
from collections import defaultdict
import time
import os

# List of 30 major cities
CITIES = [
    'New York, USA',
    'London, UK',
    'Tokyo, Japan',
    'Paris, France',
    'Beijing, China',
    'Moscow, Russia',
    'Berlin, Germany',
    'Rome, Italy',
    'Sydney, Australia',
    'Dubai, UAE',
    'Singapore, Singapore',
    'Hong Kong, China',
    'Toronto, Canada',
    'Seoul, South Korea',
    'Istanbul, Turkey',
    'Bangkok, Thailand',
    'Amsterdam, Netherlands',
    'Barcelona, Spain',
    'Vienna, Austria',
    'Stockholm, Sweden',
    'Copenhagen, Denmark',
    'Oslo, Norway',
    'Helsinki, Finland',
    'Warsaw, Poland',
    'Prague, Czech Republic',
    'Budapest, Hungary',
    'Athens, Greece',
    'Lisbon, Portugal',
    'Dublin, Ireland',
    'Brussels, Belgium'
]

def generate_city_graph(city_name):
    print(f"\nProcessing {city_name}...")
    
    try:
        # Download the road network
        G = ox.graph_from_place(city_name, network_type='drive')
        
        # Extract streets for each node
        node_to_streets = defaultdict(set)
        all_streets = set()
        
        for u, v, k, data in G.edges(keys=True, data=True):
            name = data.get('name')
            if not name:
                continue
            # Handle both list and string names
            if isinstance(name, list):
                names = [n.strip() for n in name]
            else:
                names = [name.strip()]
            for n in names:
                all_streets.add(n)
                node_to_streets[u].add(n)
                node_to_streets[v].add(n)
        
        # Build street connectivity
        street_connections = set()
        for streets in node_to_streets.values():
            streets = list(streets)
            if len(streets) > 1:
                for i in range(len(streets)):
                    for j in range(i+1, len(streets)):
                        a, b = sorted([streets[i], streets[j]])
                        street_connections.add((a, b))
        
        # Create output directory if it doesn't exist
        output_dir = '/data/qifanzhang/GLC-Benchmark/data_generation/real_world_graphs/city_road'
        os.makedirs(output_dir, exist_ok=True)
        
        # Save to file
        safe_city_name = city_name.replace(',', '_').replace(' ', '_')
        output_file = os.path.join(output_dir, f'{safe_city_name}_street_graph.txt')
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"{len(all_streets)} {len(street_connections)}\n")
            for a, b in sorted(street_connections):
                f.write(f'"{a}" "{b}"\n')
        
        print(f"Successfully generated graph for {city_name}")
        print(f"Number of streets: {len(all_streets)}")
        print(f"Number of connections: {len(street_connections)}")
        return True
        
    except Exception as e:
        print(f"Error processing {city_name}: {str(e)}")
        return False

def main():
    print("Starting to generate city graphs...")
    successful = 0
    failed = 0
    
    for city in CITIES:
        if generate_city_graph(city):
            successful += 1
        else:
            failed += 1
        # Add a small delay to avoid overwhelming the API
        time.sleep(2)
    
    print(f"\nGeneration complete!")
    print(f"Successfully processed: {successful} cities")
    print(f"Failed to process: {failed} cities")

if __name__ == "__main__":
    main() 