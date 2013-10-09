(require 'url)
(require 'json)

(defun get-alist (symbols alist)
  (if symbols
      (alist-get (cdr symbols)
                 (assoc (car symbols) alist))
    (cdr alist)))

(defun http-post (url args)
  (let ((url-request-method "POST")
        (url-request-extra-headers '(("Content-Type" . "application/x-www-form-urlencoded")))
        (url-request-data
         (mapconcat (lambda (arg)
                      (concat (url-hexify-string (car arg))
                              "="
                              (url-hexify-string (cdr arg))))
                    args
                    "&")))
    (with-current-buffer
     (url-retrieve-synchronously url)
     (goto-char (+ url-http-end-of-headers 1))
     (json-read-object))))

(defun switch-to-url-buffer (status)
  (switch-to-buffer (current-buffer)))

(defun login (username password)
  (cdaddr
   (cadar
    (http-post "http://www.reddit.com/api/login" (list (cons "api_type"  "json") (cons "user"  username) (cons "passwd"  password))))))

(defun get-comment-ids (username)
  (mapcar (lambda (x) (get-alist '(data name) x))
          (get-alist '(data children)
                     (with-current-buffer
                         (url-retrieve-synchronously (format "http://www.reddit.com/user/%s/comments.json" username))
                       (goto-char (+ url-http-end-of-headers 1))
                       (json-read-object)))))


(defun edit-comment (comment-id text modhash)
  (http-post "http://www.reddit.com/api/editusertext"
             (list (cons "api_type" "json")
                   (cons "text"  text)
                   (cons "thing_id" comment-id)
                   (cons "uh"  modhash))))

(defun edit-all-comments (username password)
  (let ((modhash (login username password)))
    (mapcar (lambda (x) (edit-comment x "Feel The Mighty Thrust Of Emacs" modhash)) (get-comment-ids username))))

(defun delete-comment (comment-id modhash)
  (http-post "http://www.reddit.com/api/del"
             (list (cons "thing_id" comment-id)
                   (cons "uh" modhash))))

(defun delete-all-comments (username password)
  (let ((modhash (login username password)))
    (mapcar (lambda (x) (delete-comment x modhash)))))



