# STORYWORLD -- EXPLORATION MODE REBUILD SPEC

**Transfer Document for Claude (JS Web-Native Refactor)**

------------------------------------------------------------------------

## 0. Project Context

StoryWorld (working title: **Gloamreach**) is a dark fantasy interactive
fiction platform.

The project previously existed in a Python/Streamlit architecture and
became overly coupled and difficult to maintain. We are now performing a
full refactor into a modular, JS-native web application.

This document reconstructs the full Exploration Mode specification that
was partially designed but never fully implemented.

This document should be treated as architectural guidance for rebuilding
Exploration Mode cleanly and modularly.

------------------------------------------------------------------------

# 1. Core Design Intent

Exploration Mode is a persistent, map-based RPG layer that:

-   Exists alongside Story Mode
-   Allows free movement between locations
-   Supports NPC interactions
-   Drives faction alignment
-   Unlocks narrative branches
-   Persists across sessions


Exploration Mode is NOT separate from Story Mode.\
It is a world-state engine that informs narrative generation.

------------------------------------------------------------------------

# 2. High-Level Player Loop

Login\
→ World Map\
→ Travel to Location\
→ NPC Interaction / Event\
→ Gain Reputation / Item / Knowledge\
→ Unlock Story Seeds\
→ Return to Map

Retention depends on: - Persistent state - Progressive unlocking -
Faction investment

------------------------------------------------------------------------

# 3. World Structure

## World Name: Gloamreach

Initial Regions (MVP supports 1 region):

1.  The Withered Vale (starter region)
2.  Blackfen Marsh
3.  Emberfall City
4.  Ironwood Frontier
5.  Pale Cathedral Ruins

Each Region Contains: - 3--6 Locations - 2--5 Persistent NPCs - 1 Hidden
Location - 1 Faction Influence

------------------------------------------------------------------------

# 4. Map System Architecture

Exploration Map is node-based, not tile-based.

Requirements:

-   Clickable region nodes
-   Location subnodes
-   Unlockable paths
-   Fog-of-war discovery
-   JSON-defined world structure

MVP should use: - Static JSON world definition - Simple interactive node
map - No heavy canvas engine required

------------------------------------------------------------------------

# 5. Persistence Model

Exploration Mode requires persistent storage.

Target stack (flexible): - PostgreSQL (Neon preferred) - Prisma or
equivalent ORM - Server-side API layer - Auth-based player identity (NOT
browser ID)

------------------------------------------------------------------------

# 6. Core Data Model

## players

-   id (uuid)
-   name
-   gender
-   alignment (hidden moral axis)
-   current_location_id
-   discovered_locations (json)
-   inventory (json)
-   faction_affinities (json)
-   reputation (json)
-   player_states (json)
-   created_at
-   last_login

## world_regions

-   id
-   name
-   description

## world_locations

-   id
-   region_id
-   name
-   description
-   is_hidden (boolean)
-   unlock_condition (json rule)

## npc_profiles

-   id
-   location_id
-   name
-   archetype
-   faction
-   disposition_base
-   memory_state (json)

## faction_reputation

-   id
-   player_id
-   faction_name
-   reputation_score
-   rank

## interaction_log

-   id
-   player_id
-   npc_id
-   choice
-   outcome
-   timestamp

------------------------------------------------------------------------

# 7. Movement System

Players can: - Travel between connected locations - Unlock hidden
locations - Be blocked by: - Reputation threshold - Required item -
Story flag - Faction alignment

Movement updates: - current_location_id - discovered_locations

------------------------------------------------------------------------

# 8. NPC Interaction Engine

Each NPC has:

-   Personality archetype
-   Faction alignment
-   Memory state (remembers player choices)
-   Disposition score

Supported interaction types:

-   Converse
-   Trade
-   Threaten
-   Investigate
-   Accept quest
-   Join faction

Dialogue generation must be constrained by: - NPC archetype -
Disposition - Player reputation - Location state - Active player states

------------------------------------------------------------------------

# 9. Faction System

Initial Factions:

-   Lantern Covenant
-   Veiled Thorn
-   Ashen Tribunal
-   Hollow Choir

Each faction has: - Ideology - Rank ladder - Reputation scale -
Exclusive quests - Cosmetic badge

Joining one faction: - Can reduce standing with others - Unlocks
faction-specific story arcs

------------------------------------------------------------------------

# 10. Reputation System

Tracked separately:

-   Global morality axis (hidden)
-   Faction reputation
-   NPC trust
-   Location reputation

Reputation affects: - Dialogue tone - Access permissions - Item
pricing - Story branches

------------------------------------------------------------------------

# 11. Inventory System (MVP)

Lightweight JSON-based inventory.

Item types: - Keys - Relics - Letters - Faction tokens - Quest items

Items can: - Unlock locations - Modify dialogue - Trigger alternate
outcomes

------------------------------------------------------------------------

# 12. Player State Effects

Possible player states:

-   Wounded
-   Blessed
-   Cursed
-   Wanted
-   Oathbound

States influence: - NPC reactions - Event frequency - Dialogue tone

------------------------------------------------------------------------

# 13. Story Mode Integration

Exploration Mode feeds Story Mode by:

-   Providing world context
-   Supplying faction alignment
-   Defining discovered lore
-   Setting relationship states

Story prompts must reference: - Current location - Active faction -
Reputation - Inventory - NPC relationships

Exploration modifies narrative logic.

------------------------------------------------------------------------

# 14. UI Layout (Conceptual)

Header: - Player name - Faction badge - Reputation indicator

Left Panel: - World Map

Center: - Location description - NPC list - Interaction buttons

Right Panel: - Inventory - Player states - Event log

------------------------------------------------------------------------

# 15. MVP Scope (Strict)

Phase 1 must include:

-   1 Region
-   3 Locations
-   3 NPCs
-   1 Faction
-   Persistent DB
-   Basic movement
-   Basic NPC conversation
-   Reputation tracking

No advanced mechanics until stable.

------------------------------------------------------------------------

# 16. Architectural Requirements

System must be modular.

Example structure:

/src\
/world\
worldData.ts\
movementEngine.ts\
reputationEngine.ts\
factionEngine.ts\
stateEngine.ts\
npcEngine.ts

/api\
players.ts\
interactions.ts

/components\
MapView.tsx\
LocationView.tsx\
NPCPanel.tsx\
InventoryPanel.tsx

/lib\
db.ts\
auth.ts

Rules:

-   No UI logic in world engine.
-   No DB logic in UI layer.
-   All state changes go through service layer.
-   World rules defined in data files, not hardcoded.

------------------------------------------------------------------------

# 17. Known Previous Failure Points

Avoid:

-   Browser-based player ID
-   Overcoupled onboarding + world logic
-   Spaghetti state mutations
-   Mixed story + exploration logic
-   Direct AI calls inside UI components

------------------------------------------------------------------------

# 18. Long-Term Expansion Goals

Future capabilities:

-   Multi-region map
-   Territory control
-   Dynamic world state
-   Seasonal events
-   Premium regions
-   Cosmetic faction upgrades
-   Companion NPC system
-   Permadeath mode

Design for extensibility.

------------------------------------------------------------------------

# 19. Development Priorities

Order of build:

1.  Auth + persistent player record
2.  World JSON structure
3.  Movement engine
4.  Location rendering
5.  NPC interaction engine
6.  Reputation + faction integration
7.  Story-mode integration hook

Do not build advanced mechanics until MVP stable.

------------------------------------------------------------------------

# 20. Primary Objective

Build a clean, extensible Exploration Mode that:

-   Persists across sessions
-   Feels like a living world
-   Integrates with narrative generation
-   Is architecturally clean
-   Can scale to investor-demo level
