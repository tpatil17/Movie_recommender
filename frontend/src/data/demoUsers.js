// Real MovieLens users from ratings_small.csv, chosen for a spread of rating
// volume and visibly different taste so switching profiles produces visibly
// different recommendations. Labels are derived from each user's top genres
// among their 4.0+ ratings; sample titles are their actual 4.5+ picks.
//
// Hardcoded rather than served from an endpoint: the set is fixed by the
// dataset, and the demo should not need another route to be up.

export const DEMO_USERS = [
  {
    id: 213,
    label: "Sci-fi & blockbusters",
    genres: "Action, Adventure, Sci-Fi",
    ratings: 910,
    samples: ["Star Wars", "Jurassic Park", "Stargate"],
  },
  {
    id: 15,
    label: "Crime & thrillers",
    genres: "Drama, Comedy, Thriller",
    ratings: 1700,
    samples: ["Se7en", "The Usual Suspects", "Get Shorty"],
  },
  {
    id: 452,
    label: "Drama & arthouse",
    genres: "Drama, Comedy, Romance",
    ratings: 1340,
    samples: ["Pulp Fiction", "Dead Man Walking", "Three Colors: Red"],
  },
  {
    id: 547,
    label: "Heavy viewer, eclectic",
    genres: "Drama, Comedy, Romance",
    ratings: 2391,
    samples: ["Taxi Driver", "Sense and Sensibility", "Leaving Las Vegas"],
  },
  {
    id: 4,
    label: "Light & mainstream",
    genres: "Comedy, Adventure, Drama",
    ratings: 204,
    samples: ["The Birdcage", "Babe", "Star Wars"],
  },
  {
    // No ratings in the dataset, so SVD has no factors for this user. Exists to
    // demonstrate the cold-start path honestly rather than hiding it.
    id: 999999,
    label: "New user (cold start)",
    genres: "No rating history",
    ratings: 0,
    samples: [],
    coldStart: true,
  },
]

export const DEFAULT_USER = DEMO_USERS[0]
