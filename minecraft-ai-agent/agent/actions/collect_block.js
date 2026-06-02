const { withTimeout } = require('./utils')

module.exports = async function collectBlock (bot, params) {
  const blockName = params.block
  const wanted = Number(params.count ?? 1)
  const positions = bot.findBlocks({
    matching: block => block && block.name === blockName,
    maxDistance: 32,
    count: wanted
  })
  const blocks = positions.map(pos => bot.blockAt(pos)).filter(Boolean)

  if (blocks.length === 0) {
    throw new Error(`No nearby ${blockName} blocks found`)
  }

  const targetBlocks = blocks.slice(0, wanted)
  await withTimeout(bot.collectBlock.collect(targetBlocks), Math.max(20000, targetBlocks.length * 15000), 'collect_block')

  return {
    ok: targetBlocks.length >= wanted,
    partial: targetBlocks.length < wanted,
    collected_attempted: targetBlocks.length,
    requested: wanted,
    message: targetBlocks.length >= wanted
      ? `Collected ${wanted} ${blockName}`
      : `Only found ${targetBlocks.length} of ${wanted} nearby ${blockName}`
  }
}
