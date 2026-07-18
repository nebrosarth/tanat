package gamedata

// quest_targets.go -- the CURATED map from each baked PvE quest (by its locale Key) to the
// mob roster indices whose death advances it. Authored by reading every quest's real journal
// text (IDS_Quest_<Key>_JournalDesc, extracted from Tanat_Data/resources.assets) and matching
// the named creature to the roster (mob_names verified against the same locale):
//
//   Демон воитель  = mobDemonRange       Демон страж      = mobDemon
//   Демон захватчик= mobDemonMeleeElite   Демон надзиратель= mobDemonRangeElite
//   Зомби крушитель= mobZombie            Зомби солдат     = mobZombieSoldier
//   Зомби губитель = mobZombieBigElite    Зомби ратоборец  = mobZombieSoldierElite
//   Голодный гуль  = mobGhoul             Одержимый гуль   = mobGhoulPossessed
//   Скелет мечник  = mobSkeleton          Скелет лучник    = mobSkeletonArcher
//   Скелет рубака  = mobSkeletonHewer     Скелет воитель   = mobSkeletonWarrior
//   Скелет берсерк = mobSkeletonBerserk   Скелет снайпер   = mobSkeletonSniper
//   Динозавр рогач = mobDinoElite         Красногривая гор.= mobGorillaElite
//   Пигмей зомби   = mobTribesmanZombie   Наместник племени= mobTribesmanBig
//   Племенной воин = mobTribesman         Племенной охотник= mobTribesmanRange
//   Голем маятник  = mobGolem             Эдилия(фея)      = mobBossFairy   Морлокай = mobBossAnhel
//
// Kept SEPARATE from the auto-generated quests_gen.go (which stays regenerable) because the
// creature mapping is a semantic reading of the journal, not a mechanical field. Every quest
// Key MUST appear here (TestEveryQuestHasTargeting); every creature target MUST actually spawn
// on the quest's map (TestQuestTargetsAreReachable), so no quest is left impossible.
//
// anyMob=true covers two cases: the two explicit «убить N любых существ/монстров» quests, and
// quests whose objective is a world interaction we do not simulate (place amulet, bring gold to
// a well, cut tam-tams, disarm a crystal, "return for your reward"...). Those name no killable
// creature, so they stay completable by fighting on the map rather than becoming stuck. Boss-
// object quests ("destroy Hekata's altar", "find the skull of one who fell to Velial") map to
// the boss they happen at, since that is the encounter the player must reach.

type questTargeting struct {
	anyMob  bool
	targets []int
}

// Family groups for quests that name a creature TYPE rather than one variant.
var (
	allSkeletons = []int{mobSkeleton, mobSkeletonArcher, mobSkeletonHewer,
		mobSkeletonWarrior, mobSkeletonBerserk, mobSkeletonSniper, mobSkeletonBurning}
	skelSwordArcher = []int{mobSkeleton, mobSkeletonArcher} // «скелеты-мечники и лучники»
	allDinos        = []int{mobDino, mobDinoElite, mobDinoRange}
	allSpiders      = []int{mobSpider, mobSpiderElite}
	allGolems       = []int{mobGolem, mobGolemElite}
	allTribes       = []int{mobTribesman, mobTribesmanRange, mobTribesmanZombie, mobTribesmanBig}
	zombies41       = []int{mobZombieBigElite, mobZombieSoldierElite} // the zombies present in map_4_1
	fourCryptBosses = []int{mobBossElgorm, mobBossVelial, mobBossCerber, mobBossHekata}
)

// kills builds a creature-targeted entry; anyKill an "any creature on the map" entry.
func kills(ids ...int) questTargeting { return questTargeting{targets: ids} }
func anyKill() questTargeting         { return questTargeting{anyMob: true} }

var questKillTargets = map[string]questTargeting{
	// ===== map_4_0 «Подземный город» =====
	"NPC1_PVE_Single_Stage1_1": anyKill(),                         // COLLECT: find Elgorm's tome in side rooms (fetch)
	"NPC1_PVE_Single_Stage1_2": anyKill(),                         // COLLECT: place amulet on the hero's grave
	"NPC1_PVE_Single_Stage1_3": kills(mobGhoul),                   // KILL 10 голодных гулей  <-- the reported bug
	"NPC1_PVE_Single_Stage1_4": kills(allSkeletons...),            // KILL 20 скелетов
	"NPC1_PVE_Single_Stage2_1": kills(mobBossElgorm),              // KILL Эльгорм
	"NPC1_PVE_Single_Stage2_2": kills(mobBossElgorm),              // COLLECT Elgorm's diary in his crystal hall
	"NPC1_PVE_Single_Stage2_3": anyKill(),                         // COLLECT a crystal from the hall (fetch)
	"NPC1_PVE_Single_Stage2_4": kills(mobGhoul),                   // KILL 25 голодных гулей
	"NPC1_PVE_Single_Stage2_5": kills(skelSwordArcher...),         // COLLECT 10 trophy weapons of sword/archer skeletons
	"NPC1_PVE_Single_Stage3_1": kills(mobBossVelial),              // KILL Велиал
	"NPC1_PVE_Single_Stage3_2": anyKill(),                         // COLLECT place an enchanted stone on the crystal
	"NPC1_PVE_Single_Stage3_3": kills(mobBossVelial),              // KILL find Osvald's skull (he fell to Velial)
	"NPC1_PVE_Single_Stage3_4": kills(mobZombieSoldier),           // COLLECT 20 zombie-soldier hides
	"NPC1_PVE_Single_Stage3_5": kills(mobZombie),                  // COLLECT 16 zombie-crusher scalps
	"NPC1_PVE_Single_Stage4_1": kills(mobBossCerber),              // KILL Цербер
	"NPC1_PVE_Single_Stage4_2": kills(mobBossCerber),              // COLLECT Cerber's chain from his room
	"NPC1_PVE_Single_Stage4_3": kills(mobZombie),                  // KILL 20 зомби-крушителей
	"NPC1_PVE_Single_Stage4_4": kills(skelSwordArcher...),         // COLLECT 15 bone powder from sword/archer skeletons
	"NPC1_PVE_Single_Stage5_1": kills(mobBossHekata),              // KILL Хеката
	"NPC1_PVE_Single_Stage5_2": anyKill(),                         // COLLECT "return for your reward" (no objective)
	"NPC1_PVE_Single_Stage5_3": kills(mobDemonRange),              // KILL 25 демонов-воителей
	"NPC1_PVE_Single_Stage5_4": kills(mobDemon),                   // COLLECT 15 axes of demon-guardians
	"NPC1_PVE_Single_Stage6_1": kills(mobBossHekata),              // KILL disarm an altar crystal by Hekata
	"NPC1_PVE_Single_Stage6_2": kills(mobBossVelial, mobBossCerber, mobBossHekata), // COLLECT their weapons
	"NPC1_PVE_Single_Stage6_3": anyKill(),                         // KILL 200 любых существ  (explicit "any")
	"NPC1_PVE_Single_Stage7_1": kills(fourCryptBosses...),         // KILL Elgorm+Velial+Cerber+Hekata
	"NPC1_PVE_Single_Stage7_2": kills(mobDemonRange),              // KILL 50 демонов-воителей
	"NPC1_PVE_Single_Stage7_3": kills(mobDemon),                   // COLLECT 30 axes of demon-guardians
	"Map_4_0_NPC1_PVE_Repeat_Stage8_1": kills(mobZombie),          // KILL 20 зомби-крушителей (replay)
	"Map_4_0_NPC1_PVE_Repeat_Stage8_2": kills(skelSwordArcher...), // KILL bone powder from sword/archer skeletons

	// ===== map_4_1 «Логово вторжения» (elite invasion) =====
	"Map_4_1_NPC1_PVE_Group_DropStage_1": anyKill(),                   // COLLECT carry a scroll (fetch)
	"Map_4_1_NPC1_PVE_Group_DropStage_2": anyKill(),                   // COLLECT bring gold to the well
	"Map_4_1_NPC1_PVE_Group_Stage1_1":    anyKill(),                   // COLLECT a magic crystal by the reborn ring
	"Map_4_1_NPC1_PVE_Group_Stage1_2":    anyKill(),                   // KILL rite of spirit-binding in the crystal hall (ritual)
	"Map_4_1_NPC1_PVE_Group_Stage1_3":    anyKill(),                   // KILL enchant the guardian statue (interaction)
	"Map_4_1_NPC1_PVE_Group_Stage1_4":    anyKill(),                   // KILL find Elgorm's cursed tome (fetch)
	"Map_4_1_NPC1_PVE_Group_Stage1_5":    kills(mobGhoulPossessed),    // KILL 10 одержимых гулей
	"Map_4_1_NPC1_PVE_Group_Stage1_6":    kills(mobSkeletonSniper),    // COLLECT 12 eyes of sniper skeletons
	"Map_4_1_NPC1_PVE_Group_Stage1_7":    kills(mobSkeletonWarrior),   // KILL 8 скелетов-воителей
	"Map_4_1_NPC1_PVE_Group_Stage1_8":    kills(mobSkeletonWarrior),   // COLLECT a key from warrior skeletons
	"Map_4_1_NPC1_PVE_Group_Stage1_9":    anyKill(),                   // KILL bring Elgorm's ring (fetch)
	"Map_4_1_NPC1_PVE_Group_Stage2_1":    anyKill(),                   // COLLECT inspect the well (interaction)
	"Map_4_1_NPC1_PVE_Group_Stage2_2":    anyKill(),                   // COLLECT bring the swords by the well (fetch)
	"Map_4_1_NPC1_PVE_Group_Stage2_3":    anyKill(),                   // COLLECT throw heads into the well (interaction)
	"Map_4_1_NPC1_PVE_Group_Stage2_4":    anyKill(),                   // COLLECT fill the lamp with oil (interaction)
	"Map_4_1_NPC1_PVE_Group_Stage2_5":    anyKill(),                   // KILL destroy the tome by using an item (interaction)
	"Map_4_1_NPC1_PVE_Group_Stage2_6":    kills(zombies41...),         // KILL find the jailer's note on a zombie
	"Map_4_1_NPC1_PVE_Group_Stage2_7":    kills(mobSkeletonBerserk),   // COLLECT 8 hearts of berserk skeletons
	"Map_4_1_NPC1_PVE_Group_Stage2_8":    kills(mobZombieSoldierElite), // COLLECT 12 shields/swords of zombie-ratoborets
	"Map_4_1_NPC1_PVE_Group_Stage2_9":    kills(mobBossVelial),        // COLLECT Velial's armour
	"Map_4_1_NPC1_PVE_Group_Stage3_1":    kills(mobBossCerber),        // KILL knock out Cerber's fangs
	"Map_4_1_NPC1_PVE_Group_Stage3_2":    anyKill(),                   // COLLECT rite of sealing (ritual)
	"Map_4_1_NPC1_PVE_Group_Stage3_3":    anyKill(),                   // COLLECT throw cooling alloy into the lava
	"Map_4_1_NPC1_PVE_Group_Stage3_4":    kills(mobDemonRangeElite),   // COLLECT jailer's pendant from a demon-overseer
	"Map_4_1_NPC1_PVE_Group_Stage3_5":    kills(mobDemonRangeElite),   // KILL 15 демонов-надзирателей
	"Map_4_1_NPC1_PVE_Group_Stage3_6":    kills(mobDemonMeleeElite),   // KILL 16 демонов-захватчиков
	"Map_4_1_NPC1_PVE_Group_Stage3_7":    kills(mobBossCerber),        // KILL Цербер
	"Map_4_1_NPC1_PVE_Group_Stage4_1":    kills(mobBossHekata),        // KILL Хеката
	"Map_4_1_NPC1_PVE_Group_Stage4_2":    kills(mobBossHekata),        // KILL destroy Hekata's altar
	"Map_4_1_NPC1_PVE_Group_Stage4_3":    kills(mobDemonMeleeElite),   // COLLECT 12 metal from demon-invaders
	"Map_4_1_NPC1_PVE_Group_Stage4_4":    anyKill(),                   // KILL 50 любых монстров (explicit "any")
	"Map_4_1_NPC1_PVE_Group_Stage4_5":    kills(mobBossHekata),        // COLLECT Hekata's necklace
	"Map_4_1_NPC1_PVE_Repeat_Stage5_1":   kills(mobSkeletonBerserk),   // COLLECT 18 hearts of berserk skeletons (replay)
	"Map_4_1_NPC1_PVE_Repeat_Stage5_2":   kills(mobDemonMeleeElite),   // KILL 25 демонов-захватчиков (replay)

	// ===== map_4_2 «Заповедные джунгли» =====
	"Map_4_2_NPC1_PVE_Single_Stage1_1":  anyKill(),                       // KILL pluck a wild flower's petal (no killable flower here)
	"Map_4_2_NPC1_PVE_Single_Stage1_2":  kills(allGolems...),             // KILL a golem by the 1st altar
	"Map_4_2_NPC1_PVE_Single_Stage1_3":  kills(allDinos...),              // KILL get dinosaur meat (purple-hided)
	"Map_4_2_NPC1_PVE_Single_Stage1_4":  kills(mobDinoElite),             // COLLECT hides of horned dinosaurs
	"Map_4_2_NPC1_PVE_Single_Stage1_5":  kills(mobTribesman),             // COLLECT weapons of tribe warriors
	"Map_4_2_NPC1_PVE_Single_Stage1_6":  kills(mobTribesmanRange),        // KILL tribe hunters
	"Map_4_2_NPC1_PVE_Single_Stage1_7":  kills(mobDinoElite),             // COLLECT horns of horned dinosaurs
	"Map_4_2_NPC1_PVE_Single_Stage1_8":  kills(allSpiders...),            // KILL spiders in the east
	"Map_4_2_NPC1_PVE_Single_Stage1_9":  anyKill(),                       // COLLECT steal rubies from snake statues
	"Map_4_2_NPC1_PVE_Single_Stage1_10": anyKill(),                       // COLLECT fix amulet on the gates
	"Map_4_2_NPC1_PVE_Single_Stage1_11": kills(mobTribesman, mobTribesmanRange), // KILL tribe warriors and hunters
	"Map_4_2_NPC1_PVE_Single_Stage1_12": anyKill(),                       // COLLECT bring offerings to the chief's hut
	"Map_4_2_NPC1_PVE_Single_Stage1_13": anyKill(),                       // COLLECT cut the tam-tams
	"Map_4_2_NPC1_PVE_Single_Stage1_14": kills(allDinos...),              // KILL 20 динозавров
	"Map_4_2_NPC1_PVE_Single_Stage1_15": kills(mobBossGrimlok),           // KILL Гримлок
	"Map_4_2_NPC1_PVE_Single_Stage1_16": kills(mobBossGrimlok),           // COLLECT Grimlok's cage
	"Map_4_2_NPC1_PVE_Single_Stage2_1":  kills(allTribes...),             // KILL collect masks from tribe shamans (no shaman mob -> tribe)
	"Map_4_2_NPC1_PVE_Single_Stage2_2":  kills(mobSkeletonBurning),       // KILL burning skeletons (the pinned «Деревенский пожар» cluster, jungleBurntVillage)
	"Map_4_2_NPC1_PVE_Single_Stage2_3":  kills(mobGorillaElite),          // KILL red gorillas
	"Map_4_2_NPC1_PVE_Single_Stage2_4":  kills(mobTribesmanZombie),       // KILL pygmy-zombies by the hill
	"Map_4_2_NPC1_PVE_Single_Stage2_5":  kills(mobBossFairy),             // KILL the fairy Edilia
	"Map_4_2_NPC1_PVE_Single_Stage2_6":  kills(mobBossFairy),             // COLLECT Edilia's braids
	"Map_4_2_NPC1_PVE_Single_Stage3_1":  anyKill(),                       // COLLECT offering at the volcano altar
	"Map_4_2_NPC1_PVE_Single_Stage3_2":  anyKill(),                       // COLLECT place rubies in the statue's eyes
	"Map_4_2_NPC1_PVE_Single_Stage3_3":  kills(mobGolem),                 // KILL pendulum golems
	"Map_4_2_NPC1_PVE_Single_Stage3_4":  kills(mobBossTitanid),           // KILL Титанид
	"Map_4_2_NPC1_PVE_Single_Stage3_5":  kills(mobBossTitanid),           // KILL destroy the altar in Titanid's temple
	"Map_4_2_NPC1_PVE_Single_Stage3_6":  kills(allGolems...),             // COLLECT crystals from golems
	"Map_4_2_NPC1_PVE_Single_Stage3_7":  kills(mobBossTitanid),           // COLLECT Titanid's heart
	"Map_4_2_NPC1_PVE_Single_Stage4_1":  anyKill(),                       // KILL seal a beast-spirit in a totem (ritual)
	"Map_4_2_NPC1_PVE_Single_Stage4_2":  kills(mobTribesmanBig, mobTribesman), // KILL viceroys and initiates of the tribe
	"Map_4_2_NPC1_PVE_Single_Stage4_3":  anyKill(),                       // KILL destroy the spirit altar (interaction)
	"Map_4_2_NPC1_PVE_Single_Stage4_4":  kills(allGolems...),             // KILL pedestal golems
	"Map_4_2_NPC1_PVE_Single_Stage4_5":  kills(mobBossAnhel),             // KILL shaman Morlokai (final boss)
	"Map_4_2_NPC1_PVE_Repeat_Stage5_1":  kills(allDinos...),              // COLLECT dinosaur meat (replay)
	"Map_4_2_NPC1_PVE_Repeat_Stage5_2":  kills(mobTribesman),             // KILL weapons of tribe warriors (replay)
	"Map_4_2_NPC1_PVE_Repeat_Stage5_3":  kills(mobTribesmanRange),        // KILL tribe hunters (replay)
	"Map_4_2_NPC1_PVE_Repeat_Stage5_4":  kills(allGolems...),             // COLLECT crystals from golems (replay)
	"Map_4_2_NPC1_PVE_Repeat_Stage5_5":  kills(mobBossAnhel),             // KILL shaman Morlokai (replay)
}
