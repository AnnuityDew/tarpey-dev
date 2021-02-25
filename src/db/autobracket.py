from instance.config import GCP_LOCATION, GCP_MM_QUEUE, GCP_PROJECT, GCP_QUEUE_SA_EMAIL
from itertools import combinations
import pandas as pd
import pathlib
from google.cloud import tasks_v2


def simulate_all_matchups():
    """Used to queue up all possible matchups for the March Madness simulator."""
    
    # need all team combos for simulation
    all_matchups_df = pd.read_csv(
        pathlib.Path(f"src/db/matchup_table_2021.csv"),
    )

    # build list of tournament teams
    away_teams = list(all_matchups_df.away_team.unique())
    home_teams = list(all_matchups_df.home_team.unique())
    # TBD is a placeholder not a team
    away_teams.remove('TBD')
    home_teams.remove('TBD')
    tournament_teams = (away_teams + home_teams)[0:5]
    tournament_matchups = list(combinations(tournament_teams, 2))
    # generate list of request URLs to queue up
    tournament_game_urls = [
        f"https://tarpey.dev/api/autobracket/sim/2020/{matchup[0]}/{matchup[1]}/5"
        for matchup in tournament_matchups
    ]

    # Create a cloud tasks client
    client = tasks_v2.CloudTasksClient()

    # google cloud task queue info
    project = GCP_PROJECT
    queue = GCP_MM_QUEUE
    location = GCP_LOCATION
    service_account_email = GCP_QUEUE_SA_EMAIL
    payload = None

    # Construct the fully qualified queue name.
    parent = client.queue_path(project, location, queue)

    # Construct the requests.
    response_list = []
    for url in tournament_game_urls:
        new_task = {
            "http_request": {
                "http_method": tasks_v2.HttpMethod.POST,
                "url": url,
                "oidc_token": {"service_account_email": service_account_email},
            }
        }
        if payload is not None:
            # The API expects a payload of type bytes.
            converted_payload = payload.encode()

            # Add the payload to the request.
            new_task["http_request"]["body"] = converted_payload

        response = client.create_task(request={"parent": parent, "task": new_task})
        print("Created task {}".format(response.name))
        response_list.append(response)

    return response_list


if __name__ == '__main__':
    simulate_all_matchups()
