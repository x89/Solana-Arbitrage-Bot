import arrow
import argparse
import json
import logging
import os
import praw
import sys
import time
import yaml
from datetime import datetime, timedelta
from praw.models import Comment, Submission
from prawcore.exceptions import ResponseException, OAuthException, BadRequest
from re import sub
from shreddit.util import get_sentence, ShredditError


class Shredder(object):
    """This class stores state for configuration, API objects, logging, etc. It exposes a shred() method that
    application code can call to start it.
    """
    def __init__(self, config, user):
        logging.basicConfig()
        self._logger = logging.getLogger("shreddit")
        self._logger.setLevel(level=logging.DEBUG if config.get("verbose", True) else logging.INFO)
        self.__dict__.update({"_{}".format(k): config[k] for k in config})

        self._user = user
        self._connect()

        if self._save_directory:
            self._r.config.store_json_result = True

        self._recent_cutoff = arrow.now().shift(hours=-self._hours)
        self._nuke_cutoff = arrow.now().shift(hours=-self._nuke_hours)
        if self._save_directory:
            if not os.path.exists(self._save_directory):
                os.makedirs(self._save_directory)

        # Add any multireddit subreddits to the whitelist
        self._whitelist = set([s.lower() for s in self._whitelist])
        for username, multiname in self._multi_whitelist:
            multireddit = self._r.multireddit(username, multiname)
            for subreddit in multireddit.subreddits:
                self._whitelist.add(str(subreddit).lower())

        # Add any multireddit subreddits to the blacklist
        self._blacklist = set()
        for username, multiname in self._multi_blacklist:
            multireddit = self._r.multireddit(username, multiname)
            for subreddit in multireddit.subreddits:
                self._blacklist.add(str(subreddit).lower())


        self._logger.info("Deleting ALL items before {}".format(self._nuke_cutoff))
        self._logger.info("Deleting items not whitelisted until {}".format(self._recent_cutoff))
        self._logger.info("Ignoring ALL items after {}".format(self._recent_cutoff))
        self._logger.info("Targeting {} sorted by {}".format(self._item, self._sort))
        if self._blacklist:
            self._logger.info("Deleting ALL items from subreddits {}".format(", ".join(list(self._blacklist))))
        if self._whitelist:
            self._logger.info("Keeping items from subreddits {}".format(", ".join(list(self._whitelist))))
        if self._keep_a_copy and self._save_directory:
            self._logger.info("Saving deleted items to: {}".format(self._save_directory))
        if self._trial_run:
            self._logger.info("Trial run - no deletion will be performed")

    def shred(self):
        deleted = self._remove_things(self._build_iterator())
        self._logger.info("Finished deleting {} items. ".format(deleted))
        if deleted >= 1000:
            # This user has more than 1000 items to handle, which angers the gods of the Reddit API. So chill for a
            # while and do it again.
            self._logger.info("Waiting {} seconds and continuing...".format(self._batch_cooldown))
            time.sleep(self._batch_cooldown)
            self._connect()
            self.shred()

    def _connect(self):
        try:
            self._r = praw.Reddit(self._user, check_for_updates=False, user_agent="python:shreddit:v6.0.4")
            self._logger.info("Logged in as {user}.".format(user=self._r.user.me()))
        except ResponseException:
            raise ShredditError("Bad OAuth credentials")
        except OAuthException:
            raise ShredditError("Bad username or password")

    def _check_whitelist(self, item):
        """Returns True if the item is whitelisted, False otherwise.
        """
        if str(item.subreddit).lower() in self._whitelist or item.id in self._whitelist_ids:
            return True
        if self._whitelist_distinguished and item.distinguished:
            return True
        if self._whitelist_gilded and item.gilded:
            return True
        if self._max_score is not None and item.score > self._max_score:
            return True
        return False

    def _save_item(self, item):
        name = item.subreddit_name_prefixed[2:]
        path = "{}/{}/{}.json".format(item.author, name, item.id)
        if not os.path.exists(os.path.join(self._save_directory, os.path.dirname(path))):
            os.makedirs(os.path.join(self._save_directory, os.path.dirname(path)))
        with open(os.path.join(self._save_directory, path), "w") as fh:
            # This is a temporary replacement for the old .json_dict property:
            output = {k: item.__dict__[k] for k in item.__dict__ if not k.startswith("_")}
            output["subreddit"] = output["subreddit"].title
            output["author"] = output["author"].name
            json.dump(output, fh, indent=2)

    def _remove_submission(self, sub):
        self._logger.info("Deleting submission: #{id} {url}".format(id=sub.id, url=sub.url.encode("utf-8")))

    def _remove_comment(self, comment):
        if self._replacement_format == "random":
            replacement_text = get_sentence()
        elif self._replacement_format == "dot":
            replacement_text = "."
        else:
            replacement_text = self._replacement_format

        short_text = sub(b"\n\r\t", " ", comment.body[:35].encode("utf-8"))
        msg = "/r/{}/ #{} ({}) with: {}".format(comment.subreddit, comment.id, short_text, replacement_text)

        self._logger.debug("Editing and deleting {msg}".format(msg=msg))
        if not self._trial_run:
            comment.edit(replacement_text)

    def _remove(self, item):
        if self._keep_a_copy and self._save_directory:
            self._save_item(item)
        if not self._trial_run:
            if self._clear_vote:
                try:
                    item.clear_vote()
                except BadRequest:
                    self._logger.debug("Couldn't clear vote on {item}".format(item=item))
        if isinstance(item, Submission):
            self._remove_submission(item)
        elif isinstance(item, Comment):
            self._remove_comment(item)
        if not self._trial_run:
            item.delete()

    def _remove_things(self, items):
        self._logger.info("Loading items to delete...")
        to_delete = [item for item in items]
        self._logger.info("Done. Starting on batch of {} items...".format(len(to_delete)))
        count, count_removed = 0, 0
        for item in to_delete:
            count += 1
            self._logger.debug("Examining item {}: {}".format(count, item))
            created = arrow.get(item.created_utc)
            if str(item.subreddit).lower() in self._blacklist:
                self._logger.debug("Deleting due to blacklist")
                count_removed += 1
                self._remove(item)
            elif created <= self._nuke_cutoff:
                self._logger.debug("Item occurs prior to nuke cutoff")
                count_removed += 1
                self._remove(item)
            elif self._check_whitelist(item):
                self._logger.debug("Skipping due to: whitelisted")
                continue
            elif created > self._recent_cutoff:
                self._logger.debug("Skipping due to: too recent")
                continue
            else:
                count_removed += 1
                self._remove(item)
        return count_removed

    def _build_iterator(self):
        item = self._r.user.me()
        if self._item == "comments":
            item = item.comments
        elif self._item == "submitted":
            item = item.submissions

        if self._sort == "new":
            return item.new(limit=None)
        elif self._sort == "top":
            return item.top(limit=None)
        elif self._sort == "hot":
            return item.hot(limit=None)
        elif self._sort == "controversial":
            return item.controversial(limit=None)
        else:
            raise ShredditError("Sorting \"{}\" not recognized.".format(self._sort))
