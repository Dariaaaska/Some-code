import os
import random
import vk_api
from vk_api.keyboard import VkKeyboard, VkKeyboardColor
from vk_api.longpoll import VkEventType, VkLongPoll

# Получаем токен из окружения сервера (если его нет — выдаст ошибку)
TOKEN = os.getenv('VK_TOKEN')
if not TOKEN:
    raise ValueError("Ошибка: не найдена переменная окружения VK_TOKEN!")

vk_session = vk_api.VkApi(token=TOKEN)
longpoll = VkLongPoll(vk_session)

# Хранилище: {user_id: {'scene': str, 'evidence': set(), 'skill': str, 'resolve': int, 'weapon': bool}}
users_db = {}

# --- СЦЕНАРИЙ И ЛОГИКА ИГРЫ ---
SCENES = {
    'skill_select': {
        'text': "Предыстория... \nВы новый наместник округа Пэн-лай, а также судья - Ди Жэньцзе, только получили своё назначение и отправляетесь в путь из столицы. В ходе игры вам предстоит раскрыть дело о скропостижной кончине вашего предшественника. \nПеред началом игры выберите ваше прошлое. Это может повлиять на ход расследования:\nВ детстве любил наблюдать за работой семейного лекаря (навык распозния ядов)\nС ранних лет учился боевому искусству (навыки самообороны)",
        'options': {'Лекарь': 'start', 'Воин': 'start'}
    },
    'start': {
        'text': "Вы только что прибыли в Пэн-лай и приступаете к своим обязанностям. Старший писец взволнованно докладывает вам, что предыдущий наместник Ван был найден мертвым в библиотеке судебной управы. С чего начнете?",
        'options': {'Осмотреть библиотеку': 'office', 'Пойти в городскую чайную собрать слухи': 'teahouse', 'Зайти в арсенал': 'armory'}
    },
    'armory': {
        'text': "Вы зашли в местный арсенал. Там среди разнообразного оружия вы выбираете стальной меч. Теперь вы вооружены.",
        'options': {'Осмотреть библиотеку': 'office', 'Пойти в городскую чайную собрать слухи': 'teahouse'}
    },
    'office': {
        'text': "Вы зашли в библиотеку. Подойдя к чайному столику вы заметили опрокинутую пиалу для чая со странным осадком на дне. Предыдущего наместника могли отравить.",
        'options': {'Надавить на Тана': 'clerk_tang', 'Изучить архивы': 'archives'},
        'evidence_granted': 'яд'
    },
    'teahouse': {
        'text': "Вы заходите в чайную. Вокруг множество посетителей, шум и гам. Пока хозяйка наливала вам чай, вы решили расспросить ее, не было ли чего необычного в последнее время, что она знает о прошлом наместнике и его служащих. Хозяйка вспомнила, что старший писец Тан вчера расплатился золотом.",
        'options': {'Идти в порт': 'docks_check', 'Изучить архивы': 'archives'},
        'evidence_granted': 'слухи_о_деньгах'
    },
    'docks_death': {
        'text': "Вы пришли в порт. На пустой пристани вас окружили бандиты. Вы безоружны и не умеете драться. Вы погибли.",
        'options': {'Начать заново': 'skill_select'}
    },
    'docks_win': {
        'text': "Вы пришли в порт. На пустой пристани на вас напали, но благодаря вашим навыкам и отменному оружию вы отбились! Вы нашли у бандитов записку с печатью Тана, в которой написаны договорённости по поводу перевоза крупных объёмов золота.",
        'options': {'Изучить архивы': 'archives', 'Идти в буддийский храм': 'temple_check'},
        'evidence_granted': 'письмо_бандитов'
    },
    'docks_win_hard': {
        'text': "Вы пришли в порт, гед на пустой пристани на вас напали. Вы безоружны, из-за чего бой дался не так легко. Вы смогли отбиться от бандитов благодаря навыкам воина, но получили неприятную рану. Ваша решимость падает на 10%.\nВы нашли у бандитов записку с печатью Тана, в которой написаны договорённости по поводу перевоза крупных объёмов золота.",
        'options': {'Изучить архивы': 'archives', 'Идти в храм': 'temple_check'},
        'evidence_granted': 'письмо_бандитов'
    },
    'clerk_tang': {
        'text': "Вы пришли к Тану и попытались на него надавить. Старший писец нервничает и клянется, что Ван умер от старости в силу почтенного возраста и нервной работы.",
        'options': {'Осмотреть библиотеку': 'office', 'Изучить архивы': 'archives'}
    },
    'archives': {
        'text': "Среди множества полок архивов вы пытаетесь предположить, где можно найти полезную информацию. Недалеко от рабочего стола вы находите непонятные записи Вана со множеством цифр. Вчитавшись в них, вы понимаете, что предыдущий наместник расследовал пропажу большого количества золота",
        'options': {'Тайно обыскать комнату Тана': 'tang_room_check', 'Идти в буддийский храм': 'temple_check'},
        'evidence_granted': 'мотив_золото'
    },
    'tang_room': {
        'text': "Вы успешно пробрались в комнату Тана и нашли дневник о контрабанде. В дневнике подробно описан учет краденого золота.",
        'options': {'Идти в буддийский храм': 'temple_check', 'Опросить слуг в управе': 'servants_check'},
        'evidence_granted': 'дневник_тана'
    },
    'temple': {
        'text': "Когда вы пришли в храм, настоятель не сразу хотел пускать вас внутрь как приверженца конфуцианства. Но всё же учтя тот факт, что вы новый наместник и судья, он хитро обернулся на стоявшего рядом монаха и лично проводил вас внутрь. Пока настоятель отошёл поговорить с монахами и принести вам чай, вы решили осмотреться. В храме вы находите тайник с золотом.",
        'options': {'Опросить слуг в управе': 'servants_check', 'Тайно обыскать комнату Тана': 'tang_room_check'},
        'evidence_granted': 'переплавленное_золото'
    },
    'caught': {
        'text': "Вас заметили! Пришлось срочно отступать. Вы потеряли 20% решимости.",
        'options': {'Изучить архивы': 'archives', 'Опросить слуг в управе': 'servants_check'}
    },
    'servants_check': {
        'text': "Слуги запуганы и не хотят говорить. Чтобы они доверились вам, нужно проявить решимость (минимальный уровень решимости: 50%).",
        'options': {'Надавить авторитетом': 'servants_talk', 'Уйти в архивы': 'archives'}
    },
    'servants_talk': {
        'text': "Ваш авторитет сработал! Слуга признался, что Тан лично принес Вану чай.",
        'options': {'Осмотреть библиотеку': 'office', 'Изучить архивы': 'archives'},
        'evidence_granted': 'показания_слуги'
    },
    'trial_start': {
        'text': "СУД. Кто убил наместника Вана?",
        'options': {'Главный писец Тан': 'accuse_tang', 'Настоятель храма': 'accuse_abbot'}
    }
}


# --- ФУНКЦИИ ВЗАИМОДЕЙСТВИЯ ---

def send_message(user_id, text, keyboard=None):
    post = {'user_id': user_id, 'message': text, 'random_id': 0}
    if keyboard: post['keyboard'] = keyboard.get_keyboard()
    vk_session.method('messages.send', post)


def create_keyboard(options, user_state):
    kb = VkKeyboard(one_time=True)
    buttons_added = 0

    for btn_text in options.keys():
        if buttons_added > 0:
            kb.add_line()
        color = VkKeyboardColor.POSITIVE if btn_text == "Начать заново" else VkKeyboardColor.PRIMARY
        kb.add_button(btn_text, color=color)
        buttons_added += 1

    # Системные кнопки добавляем отдельными рядами вниз
    if "Начать заново" not in options:
        kb.add_line()
        kb.add_button('Улики и Статус', color=VkKeyboardColor.SECONDARY)

        if user_state['resolve'] >= 60 and user_state['scene'] != 'trial_start':
            kb.add_button('Начать суд', color=VkKeyboardColor.NEGATIVE)

    return kb


def handle_trial(user_id, accusation):
    evidences = users_db[user_id]['evidence']

    if accusation == 'accuse_abbot':
        text = "\nПРОИГРЫШ. Вы обвинили Настоятеля храма, не имея против него абсолютно никаких улик. Суд счел вас некомпетентным, и вы были отстранены от должности."

    elif accusation == 'accuse_tang':
        # Словарь для красивого отображения улик в отчете
        names_map = {
            'яд': 'Яд в чашке',
            'мотив_золото': 'Документы о пропаже золота',
            'слухи_о_деньгах': 'Слухи о золотых слитках в чайной',
            'дневник_тана': 'Черновик отчета о контрабанде',
            'показания_слуги': 'Показания слуги о чае',
            'письмо_бандитов': 'Записка о перевозке золота с печатью Тана у бандитов'
        }

        report = "Итоги вашего расследования:\n\n"

        # 1. Проверяем факт убийства
        murder_proof = [names_map[e] for e in ['яд'] if e in evidences]
        if murder_proof:
            report += f"Факт отравления: есть ({', '.join(murder_proof)})\n"
        else:
            report += "Факт отравления: нет (вы не смогли опровергнуть версию о смерти в виду преклонного возраста)\n"

        # 2. Проверяем мотив
        motive_proof = [names_map[e] for e in ['мотив_золото', 'слухи_о_деньгах'] if e in evidences]
        if motive_proof:
            report += f"Мотив: есть ({', '.join(motive_proof)})\n"
        else:
            report += "Мотив: нет (суду неясно, зачем Тану убивать наместника)\n"

        # 3. Проверяем прямую связь/причастность
        link_proof = [names_map[e] for e in ['дневник_тана', 'показания_слуги', 'письмо_бандитов'] if e in evidences]
        if link_proof:
            report += f"Причастность обвиняемого: есть ({', '.join(link_proof)})\n\n"
        else:
            report += "Причастность обвиняемого: нет (нет ни одной улики, напрямую связывающей Тана с преступлением)\n\n"

        # Выносим вердикт
        if murder_proof and motive_proof and link_proof:
            text = report + "ПОБЕДА! Доказательств оказалось более чем достаточно. Прижатый к стене неопровержимыми уликами, Тан сознался в убийстве!"
        else:
            text = report + "ПРОИГРЫШ. Суд счел ваши аргументы неубедительными. Из-за нехватки доказательств дело развалилось, Тана оправдали, а вас отстранили от должности."

    # Отправляем результат и кнопку перезапуска
    kb = VkKeyboard(one_time=True)
    kb.add_button('Начать заново', color=VkKeyboardColor.POSITIVE)
    send_message(user_id, text, kb)

    # Очищаем данные игрока для новой игры
    users_db[user_id] = {'scene': 'skill_select', 'evidence': set(), 'skill': None, 'resolve': 0, 'weapon': False}




def process_message(user_id, message):
    if user_id not in users_db or message.lower() in ['начать', 'start', 'начать заново']:
        users_db[user_id] = {'scene': 'skill_select', 'evidence': set(), 'skill': None, 'resolve': 0, 'weapon': False}
        send_message(user_id, SCENES['skill_select']['text'],
                     create_keyboard(SCENES['skill_select']['options'], users_db[user_id]))
        return

    user_state = users_db[user_id]
    current_scene_id = user_state['scene']

    if message == 'Улики и Статус':
        status = f"Решимость: {user_state['resolve']}%\nНавык: {user_state['skill']}\nОружие: {'Да' if user_state['weapon'] else 'Нет'}\n\nУлики:\n"
        status += "\n".join(f"- {ev}" for ev in user_state['evidence']) if user_state['evidence'] else "Нет улик."
        send_message(user_id, status, create_keyboard(SCENES[current_scene_id]['options'], user_state))
        return

    if message == 'Начать суд' and user_state['resolve'] >= 60:
        user_state['scene'] = 'trial_start'
        send_message(user_id, SCENES['trial_start']['text'],
                     create_keyboard(SCENES['trial_start']['options'], user_state))
        return

    if current_scene_id in SCENES:
        options = SCENES[current_scene_id]['options']

        if message in options:
            next_scene_id = options[message]

            # --- ИГРОВЫЕ МЕХАНИКИ ---
            if current_scene_id == 'skill_select':
                # Адаптировано под новые названия навыков
                user_state['skill'] = 'Лекарь' if 'Лекарь' in message else 'Воин'

            if next_scene_id == 'armory':
                user_state['weapon'] = True

            elif next_scene_id == 'docks_check':
                # Если есть оружие - легкая победа (даже для лекаря)
                if user_state['weapon'] and user_state['skill'] == 'Воин':
                    next_scene_id = 'docks_win'
                # Если оружия нет, но это Воин - трудная победа (штраф)
                elif user_state['skill'] == 'Воин':
                    next_scene_id = 'docks_win_hard'
                    user_state['resolve'] = max(0, user_state['resolve'] - 10)
                # Ни оружия, ни навыка - проигрыш
                else:
                    next_scene_id = 'docks_death'

            elif next_scene_id in ['tang_room_check', 'temple_check']:
                if random.randint(1, 100) <= 40:
                    next_scene_id = 'caught'
                    user_state['resolve'] = max(0, user_state['resolve'] - 20)
                else:
                    next_scene_id = next_scene_id.replace('_check', '')

            elif next_scene_id == 'servants_talk':
                if user_state['resolve'] < 50:
                    send_message(user_id, "Слуги не поддались вашему авторитету, так как ещё не почувствовали вашей власти как нового наместника. У вас не хватает решимости (нужно 40%).",
                                 create_keyboard(SCENES['servants_check']['options'], user_state))
                    return

            elif next_scene_id.startswith('accuse_'):
                handle_trial(user_id, next_scene_id)
                return

            user_state['scene'] = next_scene_id
            next_scene = SCENES.get(next_scene_id)

            if next_scene:
                if 'evidence_granted' in next_scene:
                    evidence = next_scene['evidence_granted']
                    if evidence not in user_state['evidence']:
                        user_state['evidence'].add(evidence)

                        # 1. Считаем бонус
                        bonus = 30 if (evidence == 'яд' and user_state['skill'] == 'Лекарь') else 20
                        user_state['resolve'] = min(100, user_state['resolve'] + bonus)

                        # 2. Логика выдачи текста об уликах
                        if evidence == 'яд':
                            if user_state['skill'] == 'Лекарь':
                                send_message(user_id,
                                             f"В детстве вы не зря увлекались врачеванием. Осадок кажется вам знакомым: это точно яд. Отправка на экспертизу будет лишь формальностью. (+{bonus}% к решимости)")
                            else:
                                send_message(user_id,
                                             f"Нужна дополнительная проверка, был ли прошлый наместник отравлен. Вы отправляете пиалу на экспертизу. Вскоре лекарь подтверждает, что на дне был действительно яд. Но из-за задержки вы колеблетесь в дальнейших шагах. Найдена улика: ЯД (+{bonus}% к решимости) ❗")
                        else:
                            # 3. Стандартное сообщение для ВСЕХ остальных улик (золото, дневник и т.д.)
                            send_message(user_id, f"Найдена улика: {evidence.upper()} (+{bonus}% к решимости)")

                send_message(user_id, next_scene['text'], create_keyboard(next_scene['options'], user_state))
        else:
            send_message(user_id, "Пожалуйста, используйте кнопки на клавиатуре.", create_keyboard(options, user_state))


print("ВК Бот запущен! Ожидание сообщений...")
for event in longpoll.listen():
    if event.type == VkEventType.MESSAGE_NEW and event.to_me:
        process_message(event.user_id, event.text)