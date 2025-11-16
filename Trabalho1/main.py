import searches

def main():
    searches.bfs((94, 182), (322, 630), "100712")
    searches.astar((94, 182), (322, 630), "100712")
    searches.dfs((94, 182), (322, 630), "100712")
    searches.hill_climbing((94, 182), (322, 630), "100712")
    try:
        # searches.hill_climbing((94, 351), (581, 408), "100712") exemplo conflito hill_climbing
        searches.hill_climbing((94, 182), (322, 630), "100712")

    except Exception as error:
        print('couldnt find a path')



if __name__ == "__main__":
    main()
