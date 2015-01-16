Shreddit
###########

Details
-----------
Shreddit is a Python command line program which will take a user's post history on the website Reddit (http://reddit.com) and after having the user edit a config file will systematically go through the user's history deleting one post/submission at a time until only those whitelisted remain.
Note: When it became known that post edits were *not* saved but post deletions *were* saved code was added to edit your post prior to deletion. In fact you can actually turn off deletion all together and just have lorem ipsum (or a message about Shreddit) but this will increase how long it takes the script to run as it will be going over all of your messages every run!
Basically it lets you maintain your normal reddit account while having your history scrubbed after a certain amount of time.

Installation
-----------
The way I personally install Shreddit is via a handy tool called `virtualenv` which may come with your packagae manager or may be a part of your Python package in your distro (have a search if you can't find it). Both Python 2 and 3 are supported.

1. Clone the repository
2. Enter the repository's directory and run `virtualenv .` (this creates a virtual environment)
3. Run the following command, you must run this *every time* you wish to run the script `source ./bin/activate`
4. This installs the required modules locally to your Shreddit virtual environment `pip install -r requirements.txt`
5. Copy `shreddit.cfg.example` to something else and edit it to your liking.
6. Run `python shreddit.py -c YOUR_CONFIG_FILE.cfg`.

Alternatively try to run `./install.sh` and it will attempt to do it all for you.

Notes:

- The script *does* work with Python versions 2 and 3 but people often get in a mess with pip versions, python versions and virtulenv versions. Make sure that your Python/pip/virtualenv are all the same version. If you ran the above code it *should* work as stated.
- If in doubt try running `python3` instead of just `python` - the same goes for `pip3` and `virtualenv3` (exchange for 2 if you wish, though I advise using version 2).
- It's useful to have it run as an event, you can set this up as you like but I suggest `cron` via `crontab -e` and adding a line such as `@hourly cd $HOME/Shreddit && source bin/activate && python shreddit.py -c YOUR_CONFIG_FILE.cfg`. See below for more.
- Adding your password to the praw.ini and adding the additional output line can provide extra debugging help.

Cron examples
-----------
- Run crontab -e to edit your cron file. If you have access to something like vixie-cron then each user can have their own personal cron job!

- Run every hour on the hour
	`0 * * * * cd /home/$USER/Shreddit/ && source bin/activate && ./shreddit.py`

- Run at 3am every morning
	`0 3 * * * cd /home/$USER/Shreddit/ && source bin/activate && ./shreddit.py`

- Run once a month on the 1st of the month
	`0 0 1 * * cd /home/$USER/Shreddit/ && source bin/activate && ./shreddit.py`

Caveats
-----------
- Only your previous 1,000 comments are accessable on Reddit. So good luck deleting the others. There may be ways to hack around this via iterating using sorting by top/best/controversial/new but for now I am unsure. I believe it best to set the script settings and run it as a cron job and then it won't be a problem unless you post *a lot*. I do, however, think that it may be a caching issue and perhaps after a certain time period your post history would, once again, become available as a block of 1,000. So you needn't despair yet!

- We are relying on Reddit admin words that they do not store edits, deleted posts are still stored in the database they are merely inaccessable to the public.
