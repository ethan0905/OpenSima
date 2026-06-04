const mcDataLoader = require('minecraft-data')
const { pathfinder, Movements, goals } = require('mineflayer-pathfinder')
const collectBlock = require('mineflayer-collectblock').plugin
const pvp = require('mineflayer-pvp').plugin

const DEFAULT_BLOCKS_OF_INTEREST = [
  'oak_log',
  'birch_log',
  'spruce_log',
  'dirt',
  'grass_block',
  'stone',
  'cobblestone',
  'sand',
  'cactus',
  'coal_ore',
  'iron_ore',
  'crafting_table',
  'furnace',
  'water',
  'poppy',
  'dandelion'
]

const HOSTILE_MOBS = new Set([
  'blaze',
  'creeper',
  'drowned',
  'enderman',
  'evoker',
  'guardian',
  'husk',
  'phantom',
  'pillager',
  'ravager',
  'skeleton',
  'slime',
  'spider',
  'stray',
  'vex',
  'vindicator',
  'warden',
  'witch',
  'wither_skeleton',
  'zombie',
  'zombie_villager'
])

class MineflayerReasoningBrain {
  constructor (options = {}) {
    this.provider = options.provider || process.env.AI_PROVIDER || (process.env.GEMINI_API_KEY ? 'gemini' : 'openai')
    this.openaiApiKey = options.openaiApiKey || process.env.OPENAI_API_KEY || ''
    this.openaiModel = options.openaiModel || process.env.OPENAI_MODEL || 'gpt-4.1-mini'
    this.geminiApiKey = options.geminiApiKey || process.env.GEMINI_API_KEY || ''
    this.geminiModel = options.geminiModel || process.env.GEMINI_MODEL || 'gemini-3.1-flash-lite'
    this.blocksOfInterest = options.blocksOfInterest || DEFAULT_BLOCKS_OF_INTEREST
    this.maxDistance = Number(options.maxDistance || 32)
    this.temperature = Number(options.temperature || 0.2)
  }

  registerPlugins (bot) {
    if (!bot.pathfinder) {
      bot.loadPlugin(pathfinder)
    }
    if (!bot.collectBlock) {
      bot.loadPlugin(collectBlock)
    }
    if (!bot.pvp) {
      bot.loadPlugin(pvp)
    }

    if (bot.version && bot.pathfinder) {
      const mcData = mcDataLoader(bot.version)
      bot.pathfinder.setMovements(new Movements(bot, mcData))
    }
  }

  generateStateSnapshot (bot, humanInstruction) {
    return generateStateSnapshot(bot, humanInstruction, {
      blocksOfInterest: this.blocksOfInterest,
      maxDistance: this.maxDistance
    })
  }

  async thinkAndAct (bot, humanInstruction) {
    this.registerPlugins(bot)

    const snapshot = this.generateStateSnapshot(bot, humanInstruction)
    const decision = await this.think(snapshot)
    const execution = await this.executeDecision(bot, decision, snapshot)

    return {
      ok: execution.ok,
      snapshot,
      decision,
      execution
    }
  }

  async think (snapshot) {
    const messages = [
      {
        role: 'system',
        content: [
          'You are the reasoning brain of a Mineflayer Minecraft bot.',
          'Return only strict JSON. No markdown, no code fences, no extra text.',
          'Use the snapshot to produce a concise internal monologue and action decision.',
          'The chain_of_thought field must be a brief action-rationale checklist, not hidden private reasoning.',
          'Allowed high_level_action values: FLEE, GO_TO, COLLECT, COLLECT_BLOCK, ATTACK, CRAFT_ITEM, CHAT, CHILL.',
          'If health is low or hostile mobs are near, prioritize safety.',
          'If an instruction requires resources not visible or available, choose CHAT or CHILL and explain the blocker.',
          'JSON format:',
          JSON.stringify({
            perceptions: 'Briefly summarize what is happening based on the snapshot.',
            internal_monologue: "The bot's inner thoughts, priorities, or survival concerns.",
            chain_of_thought: [
              'Step 1: Analyze safety.',
              'Step 2: Check prerequisites for the human instruction.',
              'Step 3: Choose immediate action.'
            ],
            high_level_action: 'FLEE',
            action_parameters: {
              target: 'string block/mob/item/coordinates',
              quantity: 1
            },
            chat_message: 'What the bot will type in Minecraft chat.'
          })
        ].join('\n')
      },
      {
        role: 'user',
        content: JSON.stringify(snapshot)
      }
    ]

    const raw = this.provider === 'gemini'
      ? await this.callGemini(messages)
      : await this.callOpenAI(messages)

    return parseStrictJson(raw)
  }

  async callOpenAI (messages) {
    if (!this.openaiApiKey) {
      throw new Error('OPENAI_API_KEY is required when AI_PROVIDER=openai')
    }

    const response = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        authorization: `Bearer ${this.openaiApiKey}`,
        'content-type': 'application/json'
      },
      body: JSON.stringify({
        model: this.openaiModel,
        messages,
        temperature: this.temperature,
        response_format: { type: 'json_object' }
      })
    })

    if (!response.ok) {
      throw new Error(`OpenAI request failed: ${response.status} ${await response.text()}`)
    }

    const body = await response.json()
    return body.choices?.[0]?.message?.content || ''
  }

  async callGemini (messages) {
    if (!this.geminiApiKey) {
      throw new Error('GEMINI_API_KEY is required when AI_PROVIDER=gemini')
    }

    const systemInstruction = messages.find(message => message.role === 'system')?.content || ''
    const userContent = messages.filter(message => message.role !== 'system').map(message => message.content).join('\n')
    const url = `https://generativelanguage.googleapis.com/v1beta/models/${this.geminiModel}:generateContent?key=${this.geminiApiKey}`

    const response = await fetch(url, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        systemInstruction: {
          parts: [{ text: systemInstruction }]
        },
        contents: [
          {
            role: 'user',
            parts: [{ text: userContent }]
          }
        ],
        generationConfig: {
          temperature: this.temperature,
          responseMimeType: 'application/json'
        }
      })
    })

    if (!response.ok) {
      throw new Error(`Gemini request failed: ${response.status} ${await response.text()}`)
    }

    const body = await response.json()
    return body.candidates?.[0]?.content?.parts?.map(part => part.text || '').join('') || ''
  }

  async executeDecision (bot, decision, snapshot) {
    const action = normalizeAction(decision.high_level_action)
    const params = decision.action_parameters || {}

    if (decision.chat_message) {
      safeChat(bot, decision.chat_message)
    }

    try {
      if (action === 'FLEE') {
        return await flee(bot, snapshot)
      }

      if (action === 'GO_TO') {
        const position = parseTargetPosition(params.target)
        if (!position) throw new Error(`GO_TO requires coordinate target, got ${JSON.stringify(params.target)}`)
        bot.pathfinder.setGoal(new goals.GoalBlock(position.x, position.y, position.z))
        return { ok: true, message: `Going to ${position.x}, ${position.y}, ${position.z}` }
      }

      if (action === 'COLLECT') {
        const blockName = String(params.target || '').trim()
        if (!blockName) throw new Error('COLLECT requires action_parameters.target')
        const targetBlock = findNearestBlockByName(bot, blockName, this.maxDistance)
        if (!targetBlock) throw new Error(`No nearby ${blockName} block found`)
        await bot.collectBlock.collect(targetBlock)
        return { ok: true, message: `Collected ${blockName}` }
      }

      if (action === 'ATTACK') {
        const mobName = String(params.target || '').trim()
        const nearestMob = findNearestEntity(bot, entity => {
          if (entity.type === 'player') return false
          if (!mobName) return HOSTILE_MOBS.has(entity.name)
          return entity.name === mobName || entity.displayName === mobName
        })
        if (!nearestMob) throw new Error(`No nearby ${mobName || 'hostile mob'} found`)
        bot.pvp.attack(nearestMob)
        return { ok: true, message: `Attacking ${nearestMob.name}` }
      }

      if (action === 'CRAFT_ITEM') {
        return { ok: false, message: 'CRAFT_ITEM decision produced, but crafting execution is not implemented in this brain module yet' }
      }

      if (action === 'CHAT' || action === 'CHILL') {
        return { ok: true, message: action === 'CHAT' ? 'Sent chat message' : 'No movement action selected' }
      }

      throw new Error(`Unsupported high_level_action: ${decision.high_level_action}`)
    } catch (error) {
      safeChat(bot, `I could not execute ${action}: ${error.message}`.slice(0, 100))
      return { ok: false, error: error.message }
    }
  }
}

function generateStateSnapshot (bot, humanInstruction, options = {}) {
  const maxDistance = Number(options.maxDistance || 32)
  const blocksOfInterest = options.blocksOfInterest || DEFAULT_BLOCKS_OF_INTEREST
  const position = bot.entity?.position

  return {
    status: {
      username: bot.username,
      health: bot.health,
      food: bot.food,
      oxygen: bot.oxygenLevel ?? bot.oxygen ?? null,
      position: position ? vectorToJson(position) : null,
      held_item: bot.heldItem?.name || null,
      is_sleeping: Boolean(bot.isSleeping)
    },
    inventory: summarizeInventory(bot),
    nearby_blocks: findBlocksOfInterest(bot, blocksOfInterest, maxDistance),
    nearby_entities: findNearbyEntities(bot, maxDistance),
    humanInstruction
  }
}

function summarizeInventory (bot) {
  const summary = {}
  for (const item of bot.inventory?.items?.() || []) {
    summary[item.name] = (summary[item.name] || 0) + item.count
  }
  return Object.entries(summary)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([name, count]) => ({ name, count }))
}

function findBlocksOfInterest (bot, blockNames, maxDistance) {
  if (!bot.registry && !bot.version) return []

  const mcData = bot.registry || mcDataLoader(bot.version)
  const blocks = []

  for (const name of blockNames) {
    const blockData = mcData.blocksByName[name]
    if (!blockData) continue

    const block = bot.findBlock({
      matching: blockData.id,
      maxDistance
    })

    if (!block) continue
    blocks.push({
      name,
      position: vectorToJson(block.position),
      distance: distanceFromBot(bot, block.position)
    })
  }

  return blocks.sort((left, right) => left.distance - right.distance)
}

function findNearbyEntities (bot, maxDistance) {
  return Object.values(bot.entities || {})
    .filter(entity => entity !== bot.entity)
    .map(entity => ({
      id: entity.id,
      name: entity.name || entity.displayName || entity.username || 'unknown',
      type: entity.type,
      username: entity.username || null,
      hostile: HOSTILE_MOBS.has(entity.name),
      position: entity.position ? vectorToJson(entity.position) : null,
      distance: entity.position ? distanceFromBot(bot, entity.position) : Infinity
    }))
    .filter(entity => Number.isFinite(entity.distance) && entity.distance <= maxDistance)
    .sort((left, right) => left.distance - right.distance)
}

function normalizeAction (action) {
  const normalized = String(action || '').trim().toUpperCase()
  if (normalized === 'COLLECT_BLOCK') return 'COLLECT'
  return normalized
}

function findNearestBlockByName (bot, blockName, maxDistance) {
  const mcData = bot.registry || mcDataLoader(bot.version)
  const blockData = mcData.blocksByName[blockName]
  if (!blockData) return null
  return bot.findBlock({ matching: blockData.id, maxDistance })
}

function findNearestEntity (bot, predicate) {
  return Object.values(bot.entities || {})
    .filter(entity => entity !== bot.entity && entity.position && predicate(entity))
    .sort((left, right) => distanceFromBot(bot, left.position) - distanceFromBot(bot, right.position))[0] || null
}

async function flee (bot, snapshot) {
  const nearestEnemy = findNearestEntity(bot, entity => HOSTILE_MOBS.has(entity.name))
  if (!nearestEnemy) {
    return { ok: true, message: 'No hostile mob nearby, staying alert' }
  }

  const botPosition = bot.entity.position
  const enemyPosition = nearestEnemy.position
  const dx = botPosition.x - enemyPosition.x
  const dz = botPosition.z - enemyPosition.z
  const length = Math.sqrt(dx * dx + dz * dz) || 1
  const fleeDistance = snapshot.status.health <= 10 ? 24 : 16
  const target = botPosition.offset((dx / length) * fleeDistance, 0, (dz / length) * fleeDistance)

  bot.pathfinder.setGoal(new goals.GoalBlock(Math.floor(target.x), Math.floor(target.y), Math.floor(target.z)))
  return { ok: true, message: `Fleeing from ${nearestEnemy.name}` }
}

function parseTargetPosition (target) {
  if (!target) return null
  if (typeof target === 'object') {
    const x = Number(target.x)
    const y = Number(target.y)
    const z = Number(target.z)
    if ([x, y, z].every(Number.isFinite)) return { x: Math.floor(x), y: Math.floor(y), z: Math.floor(z) }
  }

  const parts = String(target).match(/-?\d+(?:\.\d+)?/g)
  if (!parts || parts.length < 3) return null
  const [x, y, z] = parts.map(Number)
  if (![x, y, z].every(Number.isFinite)) return null
  return { x: Math.floor(x), y: Math.floor(y), z: Math.floor(z) }
}

function vectorToJson (vector) {
  return {
    x: Number(vector.x.toFixed(2)),
    y: Number(vector.y.toFixed(2)),
    z: Number(vector.z.toFixed(2))
  }
}

function distanceFromBot (bot, position) {
  if (!bot.entity?.position || !position) return Infinity
  return Number(bot.entity.position.distanceTo(position).toFixed(2))
}

function safeChat (bot, message) {
  if (!message || typeof bot.chat !== 'function') return
  try {
    bot.chat(String(message).slice(0, 256))
  } catch (error) {
    console.warn('[reasoning-brain] Could not chat:', error.message)
  }
}

function parseStrictJson (raw) {
  const text = String(raw || '').trim()
  if (!text) throw new Error('LLM returned an empty response')

  try {
    return JSON.parse(text)
  } catch (firstError) {
    const fenced = text.match(/```(?:json)?\s*([\s\S]*?)```/i)
    const candidate = fenced ? fenced[1] : text.slice(text.indexOf('{'), text.lastIndexOf('}') + 1)
    try {
      return JSON.parse(candidate)
    } catch (secondError) {
      throw new Error(`Could not parse LLM JSON: ${firstError.message}`)
    }
  }
}

module.exports = {
  MineflayerReasoningBrain,
  generateStateSnapshot,
  parseStrictJson,
  DEFAULT_BLOCKS_OF_INTEREST
}
