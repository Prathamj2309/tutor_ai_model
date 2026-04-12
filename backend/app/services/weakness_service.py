from app.core.supabase_client import supabase

def get_weakness_report(user_id: str, subject: str = None) -> dict:
    query = supabase.table('messages') \
        .select('topic_tags, is_correct, conversations!inner(subject)') \
        .eq('user_id', user_id) \
        .eq('role', 'assistant') \
        .not_.is_('topic_tags', 'null') \
        .order('created_at', desc=True) \
        .limit(200) \
        .execute()
    
    conv_data = query.data or []
    stats = {'physics': {}, 'chemistry': {}, 'mathematics': {}, 'general': {}}
    
    for msg in conv_data:
        subj = msg.get('conversations', {}).get('subject') or 'general'
        if subj not in stats: continue
        tags = msg.get('topic_tags') or []
        for tag in tags:
            if tag not in stats[subj]:
                stats[subj][tag] = {'attempts': 0, 'errors': 0}
            stats[subj][tag]['attempts'] += 1
            if msg.get('is_correct') is False:
                stats[subj][tag]['errors'] += 1

    result = {}
    for subj_key, topics in stats.items():
        if subject and subj_key != subject: continue
        topic_list = []
        for topic, v in topics.items():
            topic_list.append({
                'topic': topic,
                'attempts': v['attempts'],
                'error_rate': v['errors'] / v['attempts'] if v['attempts'] > 0 else 0
            })
        topic_list.sort(key=lambda x: x['error_rate'], reverse=True)
        result[subj_key] = topic_list
    return result

def get_top_weak_topics(user_id: str, subject: str, n: int = 3) -> list[str]:
    report = get_weakness_report(user_id, subject)
    topics = report.get(subject, [])
    if not topics:
        fallbacks = {
            'physics': ['kinematics', 'thermodynamics'],
            'chemistry': ['equilibrium', 'bonding'],
            'mathematics': ['calculus', 'algebra'],
            'general': ['basics']
        }
        return fallbacks.get(subject, fallbacks['general'])
    return [t['topic'] for t in topics[:n]]
