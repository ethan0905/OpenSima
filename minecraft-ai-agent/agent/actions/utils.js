const { GoalNear } = require('mineflayer-pathfinder').goals

function waitForEvent (emitter, event, timeoutMs) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      cleanup()
      reject(new Error(`Timed out waiting for ${event}`))
    }, timeoutMs)

    function cleanup () {
      clearTimeout(timer)
      emitter.removeListener(event, onEvent)
    }

    function onEvent (...args) {
      cleanup()
      resolve(args)
    }

    emitter.once(event, onEvent)
  })
}

async function withTimeout (promise, timeoutMs, label) {
  let timer
  const timeout = new Promise((resolve, reject) => {
    timer = setTimeout(() => reject(new Error(`${label} timed out after ${timeoutMs}ms`)), timeoutMs)
  })
  try {
    return await Promise.race([promise, timeout])
  } finally {
    clearTimeout(timer)
  }
}

async function gotoNear (bot, position, range = 1, timeoutMs = 30000) {
  const goal = new GoalNear(position.x, position.y, position.z, range)
  await withTimeout(bot.pathfinder.goto(goal), timeoutMs, 'pathfinder.goto')
}

function findInventoryItem (bot, name) {
  return bot.inventory.items().find(item => item.name === name)
}

function sleep (ms) {
  return new Promise(resolve => setTimeout(resolve, ms))
}

module.exports = { waitForEvent, withTimeout, gotoNear, findInventoryItem, sleep }
