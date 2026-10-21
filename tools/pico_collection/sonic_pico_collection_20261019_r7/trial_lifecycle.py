"""Monitor owned processes and audit saved data even after a capture failure."""
import json
from pathlib import Path
import time



def child_failure(children):
    for child in children:
        code = child['process'].poll()
        if code is not None:
            return dict(reason='owned_process_exited', child=child['name'],
                        pid=child['process'].pid, returncode=code, log=child.get('log'))
    return None


def wait_for_trial(episode, data, children):
    deadline = time.monotonic()+180
    recording_seen = False
    while time.monotonic() < deadline:
        if (episode/'meta.json').is_file():
            return child_failure(children)
        if not recording_seen and (data/'.episode_0000.inprogress').exists():
            recording_seen = True
            deadline = time.monotonic()+60
        if (data/'_interrupted').exists():
            return dict(reason='collector_preserved_interrupted_episode')
        failure = child_failure(children)
        if failure:
            return failure
        time.sleep(.25)
    return dict(reason='trial_timeout', recording_seen=recording_seen)


def verify(path, *, allow_incomplete=False):
    # Lazy import: the robot completion path never imports the JPEG verifier.
    from verify_saved import verify as full_readback
    return full_readback(path, allow_incomplete=allow_incomplete)


def finalized_metadata(path):
    meta = json.loads((path/'meta.json').read_text())
    termination = meta.get('termination', {})
    if termination.get('writer_errors') or not (path/'frames.jsonl').is_file():
        raise ValueError('writer did not finalize cleanly')
    complete, discarded = termination.get('success') is True, meta.get('discarded') is True
    if not complete and not (meta.get('incomplete') is True):
        raise ValueError('missing finalized episode outcome')
    if discarded and (complete or meta.get('excluded_from_training') is not True):
        raise ValueError('inconsistent discard outcome')
    return dict(episode=str(path), saved_rows=meta['num_frames'], decoded_images=0,
                readback='NOT_RUN_ON_ROBOT', finalization='PASS',
                episode_complete=complete, discarded=discarded,
                timing_review='WORKSTATION_PENDING', training_export_allowed=False)


def audit_after_stop(episode, data, failure, session_result=None, *, full_readback=True):
    """Call only after owned workers have flushed/stopped, including on failure.

    No mutation of saved episodes, no restart and no silent gap filling.
    A camera failure and a valid saved episode are independent facts.
    """
    paths = ([episode] if (episode/'meta.json').is_file() else
             sorted((data/'_interrupted').glob('*/meta.json')) +
             sorted((data/'_discarded').glob('*/meta.json')))
    paths = [p.parent if p.name == 'meta.json' else p for p in paths]
    results = []
    for path in paths:
        complete = path == episode
        label = ('SAVED_EPISODE=' if complete else 'DISCARDED_EPISODE=' if path.parent.name == '_discarded'
                 else 'PRESERVED_PARTIAL_EPISODE=')
        print(label+str(path),flush=True)
        try:
            results.append(verify(path, allow_incomplete=not complete) if full_readback else finalized_metadata(path))
        except Exception as exc:
            results.append(dict(episode=str(path), readback='FAIL', error=str(exc),
                                training_export_allowed=False))
    report = dict(runtime_failure=failure, saved_data_checks=results,
                  readback_location='workstation' if not full_readback else 'current_process',
                  automatic_reconnect=False,
                  recovery='Fix inputs, then use a NEW PICO button press; never merge across disconnect.')
    outcome = (session_result or {}).get('outcome')
    intentional = (outcome in ('discarded', 'cancelled') or
                   bool(results) and all(r.get('discarded') is True for r in results))
    if intentional:
        report['outcome'] = outcome or 'discarded'
        report['excluded_from_training'] = True
        report['session_result'] = session_result
    if not paths:
        report['no_saved_episode'] = True
        report['data_root_retained'] = str(data)
    if intentional and not failure and all(
            r.get('discarded') is True and (r['readback'] in ('PASS', 'METADATA_ONLY') or r.get('finalization') == 'PASS') for r in results):
        code = 0  # Successful operator action, NOT a successful training episode.
    elif failure or not results or any((r['readback'] != 'PASS' and r.get('finalization') != 'PASS') or not r.get('episode_complete',False) for r in results):
        code = 4
    elif full_readback and any(r['timing_review'] != 'PASS' for r in results):
        code = 2
    else:
        code = 0
    report['exit_code'] = code
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    return report
