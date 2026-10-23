export interface TeamVisuals {
  name: string;
  color: string;     // HEX string
  accent: string;    // Secondary/Accent color
}

// Canonical IPL Teams and Historical Teams
// Required exact mapping:
// CSK - Yellow, MI - Blue, RCB - Red, PBKS - distinct red shade from RCB
// GT - Sky Blue (NOT gold), LSG - Light Green (NOT purple or indigo)
// KKR - Purple, DC - Dark Blue, RR - Pink, SRH - Orange
// Deccan Chargers - Grey, Kochi Tuskers Kerala - Light Purple
// Pune Warriors India - distinct sky blue, Gujarat Lions - distinct color (Amber)
// Rising Pune Supergiants - distinct color (Fuchsia)

const TEAM_MAP: Record<string, TeamVisuals> = {
  // Chennai Super Kings
  csk: { name: 'Chennai Super Kings', color: '#FACC15', accent: '#EAB308' },
  'chennai super kings': { name: 'Chennai Super Kings', color: '#FACC15', accent: '#EAB308' },

  // Mumbai Indians
  mi: { name: 'Mumbai Indians', color: '#004BA0', accent: '#00357A' },
  'mumbai indians': { name: 'Mumbai Indians', color: '#004BA0', accent: '#00357A' },

  // Royal Challengers Bangalore / Bengaluru
  rcb: { name: 'Royal Challengers Bengaluru', color: '#E32A26', accent: '#B91C1C' },
  'royal challengers bangalore': { name: 'Royal Challengers Bangalore', color: '#E32A26', accent: '#B91C1C' },
  'royal challengers bengaluru': { name: 'Royal Challengers Bengaluru', color: '#E32A26', accent: '#B91C1C' },

  // Punjab Kings / Kings XI Punjab (distinct red shade from RCB - deep crimson)
  pbks: { name: 'Punjab Kings', color: '#B91C1C', accent: '#991B1B' },
  kxip: { name: 'Kings XI Punjab', color: '#B91C1C', accent: '#991B1B' },
  'punjab kings': { name: 'Punjab Kings', color: '#B91C1C', accent: '#991B1B' },
  'kings xi punjab': { name: 'Kings XI Punjab', color: '#B91C1C', accent: '#991B1B' },

  // Gujarat Titans (SKY BLUE - strictly NOT gold)
  gt: { name: 'Gujarat Titans', color: '#0EA5E9', accent: '#0284C7' },
  'gujarat titans': { name: 'Gujarat Titans', color: '#0EA5E9', accent: '#0284C7' },

  // Lucknow Super Giants (LIGHT GREEN - strictly NOT purple or indigo)
  lsg: { name: 'Lucknow Super Giants', color: '#4ADE80', accent: '#22C55E' },
  'lucknow super giants': { name: 'Lucknow Super Giants', color: '#4ADE80', accent: '#22C55E' },

  // Kolkata Knight Riders (Purple)
  kkr: { name: 'Kolkata Knight Riders', color: '#3A225D', accent: '#581C87' },
  'kolkata knight riders': { name: 'Kolkata Knight Riders', color: '#3A225D', accent: '#581C87' },

  // Delhi Capitals / Delhi Daredevils (Dark Blue)
  dc: { name: 'Delhi Capitals', color: '#00008B', accent: '#1E3A8A' },
  dd: { name: 'Delhi Daredevils', color: '#00008B', accent: '#1E3A8A' },
  'delhi capitals': { name: 'Delhi Capitals', color: '#00008B', accent: '#1E3A8A' },
  'delhi daredevils': { name: 'Delhi Daredevils', color: '#00008B', accent: '#1E3A8A' },

  // Rajasthan Royals (Pink)
  rr: { name: 'Rajasthan Royals', color: '#EA1A85', accent: '#BE185D' },
  'rajasthan royals': { name: 'Rajasthan Royals', color: '#EA1A85', accent: '#BE185D' },

  // Sunrisers Hyderabad (Orange)
  srh: { name: 'Sunrisers Hyderabad', color: '#FF822A', accent: '#EA580C' },
  'sunrisers hyderabad': { name: 'Sunrisers Hyderabad', color: '#FF822A', accent: '#EA580C' },

  // Deccan Chargers (Grey)
  'deccan chargers': { name: 'Deccan Chargers', color: '#6A737B', accent: '#475569' },
  dcg: { name: 'Deccan Chargers', color: '#6A737B', accent: '#475569' },

  // Kochi Tuskers Kerala (Light Purple)
  'kochi tuskers kerala': { name: 'Kochi Tuskers Kerala', color: '#A855F7', accent: '#9333EA' },
  ktk: { name: 'Kochi Tuskers Kerala', color: '#A855F7', accent: '#9333EA' },

  // Pune Warriors India (Distinct sky blue - cyan-tinted lighter sky)
  'pune warriors india': { name: 'Pune Warriors India', color: '#7DD3FC', accent: '#38BDF8' },
  'pune warriors': { name: 'Pune Warriors', color: '#7DD3FC', accent: '#38BDF8' },
  pwi: { name: 'Pune Warriors India', color: '#7DD3FC', accent: '#38BDF8' },

  // Gujarat Lions (Distinct color - Amber / Lion Gold)
  'gujarat lions': { name: 'Gujarat Lions', color: '#F59E0B', accent: '#D97706' },
  gl: { name: 'Gujarat Lions', color: '#F59E0B', accent: '#D97706' },

  // Rising Pune Supergiants / Supergiant (Distinct color - Magenta / Fuchsia)
  'rising pune supergiants': { name: 'Rising Pune Supergiants', color: '#D946EF', accent: '#C026D3' },
  'rising pune supergiant': { name: 'Rising Pune Supergiant', color: '#D946EF', accent: '#C026D3' },
  rps: { name: 'Rising Pune Supergiants', color: '#D946EF', accent: '#C026D3' },
};

const DEFAULT_TEAM_COLOR: TeamVisuals = { name: 'Unknown', color: '#94a3b8', accent: '#475569' };

export const getTeamVisuals = (teamName: string | undefined | null): TeamVisuals => {
  if (!teamName) return DEFAULT_TEAM_COLOR;
  const key = teamName.trim().toLowerCase();

  // Direct match
  if (TEAM_MAP[key]) {
    return TEAM_MAP[key];
  }

  // Substring / keyword match for robustness
  if (key.includes('chennai') || key === 'csk') return TEAM_MAP.csk;
  if (key.includes('mumbai') || key === 'mi') return TEAM_MAP.mi;
  if (key.includes('bangalore') || key.includes('bengaluru') || key === 'rcb') return TEAM_MAP.rcb;
  if (key.includes('punjab') || key === 'pbks' || key === 'kxip') return TEAM_MAP.pbks;
  if (key.includes('titans') || key === 'gt') return TEAM_MAP.gt;
  if (key.includes('lucknow') || key.includes('super giants') || key === 'lsg') return TEAM_MAP.lsg;
  if (key.includes('kolkata') || key.includes('knight') || key === 'kkr') return TEAM_MAP.kkr;
  if (key.includes('delhi') || key === 'dc' || key === 'dd') return TEAM_MAP.dc;
  if (key.includes('rajasthan') || key.includes('royals') || key === 'rr') return TEAM_MAP.rr;
  if (key.includes('hyderabad') || key.includes('sunrisers') || key === 'srh') return TEAM_MAP.srh;
  if (key.includes('deccan')) return TEAM_MAP['deccan chargers'];
  if (key.includes('kochi') || key.includes('tuskers')) return TEAM_MAP['kochi tuskers kerala'];
  if (key.includes('pune warriors') || key === 'pwi') return TEAM_MAP['pune warriors india'];
  if (key.includes('rising pune')) return TEAM_MAP['rising pune supergiants'];
  if (key.includes('lions')) return TEAM_MAP['gujarat lions'];

  return DEFAULT_TEAM_COLOR;
};

export const getTeamColor = (teamName: string | undefined | null): string => {
  return getTeamVisuals(teamName).color;
};
