require('dotenv').config()

const express = require('express')
const mineflayer = require('mineflayer')
const mcDataLoader = require('minecraft-data')
const path = require('path')
const { mineflayer: mineflayerViewer } = require('prismarine-viewer')
const { pathfinder, Movements } = require('mineflayer-pathfinder')
const collectBlock = require('mineflayer-collectblock').plugin

const { getState } = require('./state')
const moveTo = require('./actions/move_to')
const followPlayer = require('./actions/follow_player')
const collectBlockAction = require('./actions/collect_block')
const craftItem = require('./actions/craft_item')
const placeBlock = require('./actions/place_block')
const buildStructure = require('./actions/build_structure')
const attackEntity = require('./actions/attack_entity')
const eatFood = require('./actions/eat_food')
const stop = require('./actions/stop')
const { waitForEvent, withTimeout } = require('./actions/utils')

const AGENT_API_PORT = Number(process.env.AGENT_API_PORT || 3001)
const MINECRAFT_HOST = process.env.MINECRAFT_HOST || 'localhost'
const MINECRAFT_PORT = Number(process.env.MINECRAFT_PORT || 25565)
const MINECRAFT_AGENT_USERNAME = process.env.MINECRAFT_AGENT_USERNAME || 'AI_Agent'
const STEP_TIMEOUT_MS = 45000
const CHAT_STATUS = process.env.AGENT_CHAT_STATUS !== 'false'
const VIEWER_PORT = Number(process.env.AGENT_VIEWER_PORT || 3002)
const VIEWER_DISTANCE = Number(process.env.AGENT_VIEWER_DISTANCE || 6)
const VIEWER_ENABLED = process.env.AGENT_VIEWER_ENABLED !== 'false'

const ACTIONS = {
  move_to: moveTo,
  follow_player: followPlayer,
  collect_block: collectBlockAction,
  craft_item: craftItem,
  place_block: placeBlock,
  build_structure: buildStructure,
  attack_entity: attackEntity,
  eat_food: eatFood,
  stop
}

let isSpawned = false
let isExecuting = false

const bot = mineflayer.createBot({
  host: MINECRAFT_HOST,
  port: MINECRAFT_PORT,
  username: MINECRAFT_AGENT_USERNAME
})

bot.loadPlugin(pathfinder)
bot.loadPlugin(collectBlock)

bot.once('spawn', () => {
  isSpawned = true
  const mcData = mcDataLoader(bot.version)
  bot.pathfinder.setMovements(new Movements(bot, mcData))
  startViewer()
  console.log(`[agent] Spawned as ${bot.username} on ${MINECRAFT_HOST}:${MINECRAFT_PORT}`)
  sayStatus('online and ready')
})

bot.on('end', () => {
  isSpawned = false
  console.log('[agent] Disconnected from Minecraft server')
})

bot.on('kicked', reason => console.log('[agent] Kicked:', reason))
bot.on('error', error => console.error('[agent] Error:', error.message))
bot.on('health', () => console.log(`[agent] Health=${bot.health} Food=${bot.food}`))

const app = express()
app.use(express.json({ limit: '128kb' }))

app.get('/dashboard', (req, res) => {
  res.sendFile(path.join(__dirname, 'dashboard.html'))
})

app.get('/dashboard-config', (req, res) => {
  res.json({
    viewer_enabled: VIEWER_ENABLED,
    viewer_url: `http://localhost:${VIEWER_PORT}`
  })
})

app.get('/state', (req, res) => {
  try {
    res.json(getState(bot, isSpawned, isExecuting))
  } catch (error) {
    res.status(500).json({ ok: false, error: error.message })
  }
})

app.post('/action-plan', async (req, res) => {
  if (isExecuting) {
    return res.status(409).json({ ok: false, error: 'Agent is already executing an action plan' })
  }

  const validationError = validateActionPlan(req.body)
  if (validationError) {
    return res.status(400).json({ ok: false, error: validationError })
  }

  isExecuting = true
  try {
    if (!isSpawned) {
      await waitForEvent(bot, 'spawn', 30000)
    }
    const result = await executePlan(req.body)
    res.json(result)
  } catch (error) {
    res.status(500).json({ ok: false, error: error.message, completed_steps: [] })
  } finally {
    isExecuting = false
  }
})

async function executePlan (actionPlan) {
  const completedSteps = []
  const stepResults = []

  for (const step of actionPlan.plan) {
    const handler = ACTIONS[step.action]
    try {
      console.log(`[agent] Step ${step.step_id}: ${step.action}`, step.params)
      sayStatus(`starting ${step.action}`)
      const result = await withTimeout(handler(bot, step.params || {}), STEP_TIMEOUT_MS, `step ${step.step_id}`)
      const normalizedResult = { step_id: step.step_id, action: step.action, ...result }
      completedSteps.push(step.step_id)
      stepResults.push(normalizedResult)
      sayStatus(result?.partial ? `partial ${step.action}` : `done ${step.action}`)

      if (result && result.ok === false && !result.partial) {
        return {
          ok: false,
          failed_step_id: step.step_id,
          error: result.message || `${step.action} failed`,
          completed_steps: completedSteps,
          step_results: stepResults
        }
      }
    } catch (error) {
      console.error(`[agent] Step ${step.step_id} failed:`, error.message)
      sayStatus(`failed ${step.action}`)
      return {
        ok: false,
        failed_step_id: step.step_id,
        error: error.message,
        completed_steps: completedSteps,
        step_results: stepResults
      }
    }
  }

  return {
    ok: true,
    completed_steps: completedSteps,
    step_results: stepResults
  }
}

function sayStatus (message) {
  if (!CHAT_STATUS || !isSpawned) return
  try {
    bot.chat(`[AI] ${message}`.slice(0, 100))
  } catch (error) {
    console.warn('[agent] Could not send chat status:', error.message)
  }
}

function startViewer () {
  if (!VIEWER_ENABLED) return
  try {
    mineflayerViewer(bot, {
      port: VIEWER_PORT,
      firstPerson: true,
      viewDistance: VIEWER_DISTANCE
    })
    console.log(`[viewer] Agent view available at http://localhost:${VIEWER_PORT}`)
  } catch (error) {
    console.warn('[viewer] Could not start Prismarine Viewer:', error.message)
  }
}

function validateActionPlan (body) {
  if (!body || typeof body !== 'object') return 'Body must be a JSON object'
  if (typeof body.goal !== 'string' || !body.goal.trim()) return 'goal is required'
  if (typeof body.message_to_user !== 'string' || !body.message_to_user.trim()) return 'message_to_user is required'
  if (!Array.isArray(body.plan) || body.plan.length === 0 || body.plan.length > 12) return 'plan must contain 1 to 12 steps'

  const suspicious = JSON.stringify(body).toLowerCase()
  for (const token of ['/give', '/tp', '/op', '/execute', '/summon', '/kill', '/setblock', '/fill', 'servercommand']) {
    if (suspicious.includes(token)) return 'Plan contains suspicious command-like content'
  }

  for (const step of body.plan) {
    if (!step || typeof step !== 'object') return 'Each step must be an object'
    if (typeof step.step_id !== 'string' || !step.step_id.trim()) return 'Each step requires step_id'
    if (!Object.prototype.hasOwnProperty.call(ACTIONS, step.action)) return `Unknown action: ${step.action}`
    if (!step.params || typeof step.params !== 'object' || Array.isArray(step.params)) return `${step.action} params must be an object`
    if (step.action !== 'stop' && Object.keys(step.params).length === 0) return `${step.action} requires params`
  }

  return null
}

app.listen(AGENT_API_PORT, () => {
  console.log(`[api] Mineflayer agent API listening on http://localhost:${AGENT_API_PORT}`)
  console.log(`[dashboard] Open http://localhost:${AGENT_API_PORT}/dashboard`)
})
